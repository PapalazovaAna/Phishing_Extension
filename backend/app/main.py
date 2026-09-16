from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.ml.predictor import predict_url
from app.reputation import check_reputation


app = FastAPI(
    title="Phishing Detection API",
    version="1.0.0"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalysisPart(BaseModel):
    score: int = Field(default=0, ge=0, le=100)
    indicators: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    url: str
    url_analysis: AnalysisPart
    dom: AnalysisPart


def classify_score(score: int) -> str:
    if score >= 70:
        return "Likely phishing"

    if score >= 40:
        return "Suspicious"

    return "Likely safe"


def combine_indicators(
    url_indicators: list[str],
    ml_indicator: str,
    dom_indicators: list[str],
    reputation_indicator: str | None,
) -> list[str]:
    indicators: list[str] = []

    indicators.extend(
        f"URL: {indicator}"
        for indicator in url_indicators
    )

    indicators.append(f"ML: {ml_indicator}")

    indicators.extend(
        f"DOM: {indicator}"
        for indicator in dom_indicators
    )

    if reputation_indicator:
        indicators.append(
            f"Reputation: {reputation_indicator}"
        )

    return indicators


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok"
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict[str, Any]:
    ml_result = predict_url(request.url)
    reputation_result = check_reputation(request.url)

    url_score = request.url_analysis.score
    ml_score = ml_result["score"]
    dom_score = request.dom.score
    reputation_score = reputation_result.get("score")
    reputation_status = reputation_result.get("status")

    has_strong_dom_indicator = any(
        "another origin" in indicator.lower()
        or "password" in indicator.lower()
        for indicator in request.dom.indicators
    )

    has_strong_url_indicator = url_score >= 60

    if reputation_status == "malicious":
        final_score = (
            url_score * 0.25
            + ml_score * 0.35
            + dom_score * 0.20
            + reputation_score * 0.20
        )

        final_score = max(final_score, 85)

    elif reputation_status == "trusted":
        final_score = (
            url_score * 0.25
            + ml_score * 0.10
            + dom_score * 0.15
        )

        if not has_strong_url_indicator and not has_strong_dom_indicator:
            final_score = min(final_score, 20)

    else:
        final_score = (
            url_score * 0.35
            + ml_score * 0.45
            + dom_score * 0.20
        )

    final_score = round(
        min(max(final_score, 0), 100)
    )

    if (
        reputation_status == "trusted"
        and final_score < 40
    ):
        ml_indicator = (
            "The URL model detected a suspicious pattern, "
            "but the domain is trusted."
        )
    else:
        ml_indicator = ml_result["indicator"]

    indicators = combine_indicators(
        url_indicators=request.url_analysis.indicators,
        ml_indicator=ml_indicator,
        dom_indicators=request.dom.indicators,
        reputation_indicator=reputation_result.get("indicator"),
    )

    return {
        "score": final_score,
        "classification": classify_score(final_score),
        "indicators": indicators,
        "components": {
            "url_score": url_score,
            "ml_score": ml_score,
            "dom_score": dom_score,
            "reputation_score": reputation_score,
        },
        "ml": {
            "score": ml_result["score"],
            "classification": ml_result["classification"],
            "probability": ml_result["probability"],
        },
        "reputation": reputation_result,
    }