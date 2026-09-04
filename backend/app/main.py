from urllib.parse import urlparse

from fastapi import FastAPI
from pydantic import BaseModel, Field


app = FastAPI(
    title="Phishing Risk Detector API",
    version="0.1.0",
)


class DomFeatures(BaseModel):
    score: float = Field(default=0, ge=0, le=100)
    indicators: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    url: str = Field(min_length=1)
    dom: DomFeatures = Field(default_factory=DomFeatures)


def analyze_url(url: str) -> dict:
    score = 0
    indicators = []

    parsed_url = urlparse(url)

    if parsed_url.scheme != "https":
        score += 10
        indicators.append("The page does not use HTTPS.")

    hostname = parsed_url.hostname or ""
    hostname_parts = hostname.split(".")

    is_ip_address = len(hostname_parts) == 4 and all(
        part.isdigit() and 0 <= int(part) <= 255
        for part in hostname_parts
    )

    if is_ip_address:
        score += 25
        indicators.append("The hostname is an IP address.")

    if len(url) > 100:
        score += 10
        indicators.append("The URL is unusually long.")

    if len(hostname_parts) > 3:
        score += 10
        indicators.append("The hostname contains many subdomains.")

    suspicious_words = [
        "login",
        "verify",
        "secure",
        "account",
        "password",
    ]

    lower_url = url.lower()

    found_word = next(
        (
            word
            for word in suspicious_words
            if word in lower_url
        ),
        None,
    )

    if found_word:
        score += 15
        indicators.append(
            f'The URL contains the suspicious term "{found_word}".'
        )

    return {
        "score": min(score, 100),
        "indicators": indicators,
    }


def classify(score: float) -> str:
    if score >= 60:
        return "Dangerous"

    if score >= 20:
        return "Suspicious"

    return "Likely safe"


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok"
    }


@app.post("/analyze")
def analyze(request: AnalyzeRequest) -> dict:
    url_result = analyze_url(request.url)

    final_score = round(
        url_result["score"] * 0.6
        + request.dom.score * 0.4
    )

    indicators = [
        f"URL: {item}"
        for item in url_result["indicators"]
    ] + [
        f"DOM: {item}"
        for item in request.dom.indicators
    ]

    return {
        "score": final_score,
        "classification": classify(final_score),
        "indicators": indicators,
        "components": {
            "url_score": url_result["score"],
            "dom_score": request.dom.score,
        },
    }