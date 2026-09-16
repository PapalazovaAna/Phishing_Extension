from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def create_request(
    url: str,
    url_score: int = 0,
    url_indicators: list[str] | None = None,
    dom_score: int = 0,
    dom_indicators: list[str] | None = None,
) -> dict:
    return {
        "url": url,
        "url_analysis": {
            "score": url_score,
            "indicators": url_indicators or [],
        },
        "dom": {
            "score": dom_score,
            "indicators": dom_indicators or [],
        },
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok"
    }


def test_trusted_domain_has_low_risk():
    request_data = create_request(
        url="https://www.google.com/"
    )

    response = client.post(
        "/analyze",
        json=request_data
    )

    assert response.status_code == 200

    result = response.json()

    assert result["score"] <= 20
    assert result["classification"] == "Likely safe"
    assert result["reputation"]["status"] == "trusted"


def test_malicious_domain_has_high_risk():
    request_data = create_request(
        url="http://phishing.test/login",
        url_score=40,
        url_indicators=[
            "The page does not use HTTPS.",
            "The URL contains the suspicious term \"login\".",
        ],
        dom_score=10,
        dom_indicators=[
            "The page contains a password input field."
        ],
    )

    response = client.post(
        "/analyze",
        json=request_data
    )

    assert response.status_code == 200

    result = response.json()

    assert result["score"] >= 85
    assert result["classification"] == "Likely phishing"
    assert result["reputation"]["status"] == "malicious"


def test_unknown_domain_is_analyzed():
    request_data = create_request(
        url="http://unknown-domain-example.com/login",
        url_score=25,
        url_indicators=[
            "The page does not use HTTPS.",
            "The URL contains the suspicious term \"login\".",
        ],
        dom_score=10,
        dom_indicators=[
            "The page contains a password input field."
        ],
    )

    response = client.post(
        "/analyze",
        json=request_data
    )

    assert response.status_code == 200

    result = response.json()

    assert 0 <= result["score"] <= 100
    assert result["classification"] in [
        "Likely safe",
        "Suspicious",
        "Likely phishing",
    ]
    assert result["reputation"]["status"] == "unknown"


def test_response_contains_all_components():
    request_data = create_request(
        url="https://example.com/"
    )

    response = client.post(
        "/analyze",
        json=request_data
    )

    assert response.status_code == 200

    result = response.json()

    assert "score" in result
    assert "classification" in result
    assert "indicators" in result
    assert "components" in result
    assert "ml" in result
    assert "reputation" in result

    assert "url_score" in result["components"]
    assert "ml_score" in result["components"]
    assert "dom_score" in result["components"]
    assert "reputation_score" in result["components"]


def test_dom_indicators_increase_risk_for_unknown_domain():
    request_data = create_request(
        url="http://unknown-login-example.com/",
        url_score=25,
        dom_score=40,
        dom_indicators=[
            "The page contains a password input field.",
            "A form submits data to another origin.",
        ],
    )

    response = client.post(
        "/analyze",
        json=request_data
    )

    assert response.status_code == 200

    result = response.json()

    assert result["score"] > 20
    assert any(
        indicator.startswith("DOM:")
        for indicator in result["indicators"]
    )