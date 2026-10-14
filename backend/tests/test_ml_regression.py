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


def test_google_remains_safe():
    response = client.post(
        "/analyze",
        json=create_request(
            "https://www.google.com/"
        ),
    )

    assert response.status_code == 200

    result = response.json()

    assert result["score"] <= 20
    assert result["classification"] == "Likely safe"


def test_gmail_remains_safe_despite_ml_signal():
    response = client.post(
        "/analyze",
        json=create_request(
            "https://mail.google.com/"
            "mail/u/0/#inbox"
        ),
    )

    assert response.status_code == 200

    result = response.json()

    assert result["score"] <= 20
    assert result["classification"] == "Likely safe"


def test_booking_long_url_remains_safe():
    response = client.post(
        "/analyze",
        json=create_request(
            "https://www.booking.com/"
            "searchresults.html?"
            "si=ai%2Cco%2Cci%2Cre%2Cla%2Cdi;"
            "dest_type=hotel;"
            "dest_id=8767025;"
            "ifl=1;"
            "aid=304142;"
            "label=gen173nr-10CBcoggI46AdIM1gEaJMBiAEBmAEzuAEXyAEM2AED6AEB-AEBiAIBqAIBuAKQn9zVBsACAdICJGFiZjBjOGE2LTNhZTMtNDkyNi1iY2Y5LWI0ZTc5OTI4NzQ0YdgCAeACAQ"
        ),
    )

    assert response.status_code == 200

    result = response.json()

    assert result["score"] <= 20
    assert result["classification"] == "Likely safe"


def test_typosquatting_domain_is_suspicious():
    response = client.post(
        "/analyze",
        json=create_request(
            "https://rnicrosoft.com/"
        ),
    )

    assert response.status_code == 200

    result = response.json()

    assert result["score"] >= 40
    assert result["classification"] == "Suspicious"

    assert result["reputation"]["similar_domain"] == "microsoft.com"
    assert any(
        indicator.startswith("Reputation:")
        and "microsoft.com" in indicator
        for indicator in result["indicators"]
    )


def test_suspicious_login_url_is_phishing():
    response = client.post(
        "/analyze",
        json=create_request(
            "http://secure-login-verification.example/login",
            url_score=25,
            url_indicators=[
                "The page does not use HTTPS.",
                'The URL contains the suspicious term "login".',
            ],
            dom_score=10,
            dom_indicators=[
                "The page contains a password input field.",
            ],
        ),
    )

    assert response.status_code == 200

    result = response.json()

    assert result["score"] >= 70
    assert result["classification"] == "Likely phishing"


def test_long_url_does_not_return_server_error():
    response = client.post(
        "/analyze",
        json=create_request(
            "https://www.booking.com/"
            "searchresults.html?"
            "si=ai%2Cco%2Cci%2Cre%2Cla%2Cdi;"
            "dest_type=hotel;"
            "dest_id=8767025;"
            "ifl=1;"
            "aid=304142;"
            "label=gen173nr-10CBcoggI46AdIM1gEaJMBiAEBmAEzuAEXyAEM2AED6AEB-AEBiAIBqAIBuAKQn9zVBsACAdICJGFiZjBjOGE2LTNhZTMtNDkyNi1iY2Y5LWI0ZTc5OTI4NzQ0YdgCAeACAQ"
        ),
    )

    assert response.status_code == 200