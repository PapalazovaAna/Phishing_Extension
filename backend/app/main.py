from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.predictor import predict_url
from app.reputation import (
    check_reputation,
    load_reputation_data,
)
from app.scoring import (
    SUSPICIOUS_THRESHOLD,
    classify_score,
    combine_scores,
    should_block,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the reputation lists and typosquatting index at startup
    # instead of during the first request.
    load_reputation_data()
    yield


app = FastAPI(
    title="Phishing Detection API",
    version="1.0.0",
    lifespan=lifespan,
)


# Only Chrome extensions may call the API from a browser, so that
# web pages cannot send requests to the local backend.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"chrome-extension://[a-p]{32}",
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# Limits for the size of a request.
MAX_URL_LENGTH = 8192
MAX_INDICATORS = 50
MAX_INDICATOR_LENGTH = 500

Indicator = Annotated[str, Field(max_length=MAX_INDICATOR_LENGTH)]


class AnalysisPart(BaseModel):
    score: int = Field(default=0, ge=0, le=100)
    indicators: list[Indicator] = Field(
        default_factory=list,
        max_length=MAX_INDICATORS,
    )


class AnalyzeRequest(BaseModel):
    url: str = Field(min_length=1, max_length=MAX_URL_LENGTH)
    url_analysis: AnalysisPart
    dom: AnalysisPart


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


def create_ml_indicator(
    ml_result: dict[str, Any],
    reputation_status: str,
    final_score: int,
) -> str:
    if (
        reputation_status == "popular"
        and ml_result["score"] >= SUSPICIOUS_THRESHOLD
        and final_score < SUSPICIOUS_THRESHOLD
    ):
        return (
            "The URL model detected a suspicious pattern "
            f"({ml_result['score']}%), but the domain is "
            "in the Tranco popular domain list."
        )

    return ml_result["indicator"]


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
    reputation_status = reputation_result["status"]
    reputation_score = reputation_result["score"] or 0

    final_score = combine_scores(
        url_score=url_score,
        ml_score=ml_score,
        dom_score=dom_score,
        reputation_status=reputation_status,
        reputation_score=reputation_score,
    )

    indicators = combine_indicators(
        url_indicators=request.url_analysis.indicators,
        ml_indicator=create_ml_indicator(
            ml_result,
            reputation_status,
            final_score,
        ),
        dom_indicators=request.dom.indicators,
        reputation_indicator=reputation_result.get("indicator"),
    )

    return {
        "score": final_score,
        "classification": classify_score(final_score),
        "block": should_block(
            final_score=final_score,
            url_score=url_score,
            ml_score=ml_score,
            reputation_status=reputation_status,
        ),
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
