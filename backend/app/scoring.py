"""Combination of the individual analysis scores into one risk score.

Every component score is between 0 and 100:

- url:        heuristic URL analysis (computed by the extension)
- ml:         URLBERT phishing probability * 100
- dom:        heuristic page content analysis (computed by the extension)
- reputation: OpenPhish / Tranco / typosquatting result

The weights depend on the reputation status, because a match in a
phishing feed or a popular domain is much stronger evidence than
the other signals.
"""

PHISHING_THRESHOLD = 70
SUSPICIOUS_THRESHOLD = 40

# Weights per reputation status, each row sums to 1.
# For popular domains the reputation score is 0, so its weight
# pulls the final score down.
WEIGHTS = {
    "malicious": {
        "url": 0.25,
        "ml": 0.35,
        "dom": 0.15,
        "reputation": 0.25,
    },
    "suspicious": {
        "url": 0.20,
        "ml": 0.25,
        "dom": 0.10,
        "reputation": 0.45,
    },
    "popular": {
        "url": 0.25,
        "ml": 0.10,
        "dom": 0.15,
        "reputation": 0.50,
    },
    "unknown": {
        "url": 0.35,
        "ml": 0.45,
        "dom": 0.20,
        "reputation": 0.0,
    },
}

# Minimum final score for reputation matches.
MALICIOUS_MIN_SCORE = 85
SUSPICIOUS_MIN_SCORE = 50

# Popular domains are capped at this score unless the URL or
# page content analysis is itself suspicious.
POPULAR_MAX_SCORE = 20

# Thresholds above which a component counts as a strong signal.
STRONG_URL_SCORE = 20
STRONG_ML_SCORE = 70
STRONG_DOM_SCORE = 10

# When this many components are strong signals at once,
# the page is at least "Likely phishing".
STRONG_SIGNALS_FOR_PHISHING = 3

# Additional rules for showing the blocking warning page.
BLOCK_ML_SCORE = 90


def classify_score(score: int) -> str:
    if score >= PHISHING_THRESHOLD:
        return "Likely phishing"

    if score >= SUSPICIOUS_THRESHOLD:
        return "Suspicious"

    return "Likely safe"


def count_strong_signals(
    url_score: int,
    ml_score: int,
    dom_score: int,
    reputation_status: str,
) -> int:
    strong_signals = [
        url_score >= STRONG_URL_SCORE,
        ml_score >= STRONG_ML_SCORE,
        dom_score >= STRONG_DOM_SCORE,
        reputation_status in {"malicious", "suspicious"},
    ]

    return sum(strong_signals)


def combine_scores(
    url_score: int,
    ml_score: int,
    dom_score: int,
    reputation_status: str,
    reputation_score: int,
) -> int:
    weights = WEIGHTS.get(
        reputation_status,
        WEIGHTS["unknown"],
    )

    final_score = (
        url_score * weights["url"]
        + ml_score * weights["ml"]
        + dom_score * weights["dom"]
        + reputation_score * weights["reputation"]
    )

    if reputation_status == "malicious":
        final_score = max(final_score, MALICIOUS_MIN_SCORE)

    elif reputation_status == "suspicious":
        final_score = max(final_score, SUSPICIOUS_MIN_SCORE)

    elif reputation_status == "popular":
        if (
            url_score < SUSPICIOUS_THRESHOLD
            and dom_score < SUSPICIOUS_THRESHOLD
        ):
            final_score = min(final_score, POPULAR_MAX_SCORE)

    final_score = round(min(max(final_score, 0), 100))

    strong_signal_count = count_strong_signals(
        url_score=url_score,
        ml_score=ml_score,
        dom_score=dom_score,
        reputation_status=reputation_status,
    )

    if strong_signal_count >= STRONG_SIGNALS_FOR_PHISHING:
        final_score = max(final_score, PHISHING_THRESHOLD)

    return final_score


def should_block(
    final_score: int,
    url_score: int,
    ml_score: int,
    reputation_status: str,
) -> bool:
    """Whether the extension shows the full-page warning
    before the user interacts with the page."""

    if reputation_status == "malicious":
        return True

    if final_score >= PHISHING_THRESHOLD:
        return True

    if ml_score >= BLOCK_ML_SCORE and url_score >= STRONG_URL_SCORE:
        return True

    return reputation_status == "suspicious" and (
        ml_score >= STRONG_ML_SCORE
        or url_score >= STRONG_URL_SCORE
    )
