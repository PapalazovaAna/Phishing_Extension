import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.reputation import load_malicious_urls
from app.scoring import (
    WEIGHTS,
    classify_score,
    combine_scores,
    should_block,
)


client = TestClient(app)


def test_weights_sum_to_one():
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "score, classification",
    [
        (0, "Likely safe"),
        (39, "Likely safe"),
        (40, "Suspicious"),
        (69, "Suspicious"),
        (70, "Likely phishing"),
        (100, "Likely phishing"),
    ],
)
def test_classify_score(score, classification):
    assert classify_score(score) == classification


def test_malicious_reputation_has_minimum_score():
    assert combine_scores(0, 0, 0, "malicious") == 85


@pytest.mark.parametrize(
    "url_score, ml_score, dom_score, expected",
    [
        # weighted 20 + 40 + 10 = 70, raised to the minimum
        (400 / 7, 400 / 9, 50, 85),
        # weighted 15 + 30 + 5 = 50, raised to the minimum
        (300 / 7, 200 / 3, 25, 85),
        # weighted 30 + 45 + 17 = 92, already above the minimum
        (600 / 7, 100, 85, 92),
    ],
)
def test_malicious_minimum_only_raises_lower_scores(
    url_score,
    ml_score,
    dom_score,
    expected,
):
    assert combine_scores(
        url_score,
        ml_score,
        dom_score,
        "malicious",
    ) == expected


def test_suspicious_reputation_has_minimum_score():
    assert combine_scores(0, 0, 0, "suspicious") == 50


def test_popular_domain_is_capped_despite_ml_score():
    assert combine_scores(10, 100, 0, "popular") == 20


def test_popular_domain_is_not_capped_with_suspicious_page():
    assert combine_scores(80, 100, 80, "popular") > 20


def test_unknown_domain_uses_url_ml_and_dom():
    assert combine_scores(20, 50, 10, "unknown") == round(
        20 * 0.35 + 50 * 0.45 + 10 * 0.20
    )


def test_three_strong_signals_are_likely_phishing():
    assert combine_scores(20, 70, 10, "unknown") == 70


def test_same_weights_for_every_reputation_status():
    # Without a rule that applies, the reputation status does not
    # change the weighted score.
    assert combine_scores(10, 20, 0, "unknown") == combine_scores(
        10,
        20,
        0,
        "popular",
    )


@pytest.mark.parametrize(
    "final_score, url_score, ml_score, status, expected",
    [
        (85, 0, 0, "malicious", True),
        (70, 0, 0, "unknown", True),
        (60, 20, 90, "unknown", True),
        (50, 0, 70, "suspicious", True),
        (50, 20, 0, "suspicious", True),
        (50, 0, 0, "suspicious", False),
        (69, 10, 95, "unknown", False),
        (10, 0, 0, "popular", False),
    ],
)
def test_should_block(
    final_score,
    url_score,
    ml_score,
    status,
    expected,
):
    assert should_block(
        final_score,
        url_score,
        ml_score,
        status,
    ) is expected


def test_api_returns_block_decision():
    phishing_url = sorted(load_malicious_urls())[0]

    response = client.post(
        "/analyze",
        json={
            "url": phishing_url,
            "url_analysis": {"score": 0, "indicators": []},
            "dom": {"score": 0, "indicators": []},
        },
    )

    assert response.status_code == 200
    assert response.json()["block"] is True


def test_api_does_not_block_popular_domain():
    response = client.post(
        "/analyze",
        json={
            "url": "https://www.google.com/",
            "url_analysis": {"score": 0, "indicators": []},
            "dom": {"score": 0, "indicators": []},
        },
    )

    assert response.status_code == 200
    assert response.json()["block"] is False
