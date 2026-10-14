from pathlib import Path

from app.reputation import (
    DATA_DIR,
    check_reputation,
    extract_registered_domain,
    load_malicious_urls,
)


def test_extract_google_registered_domain():
    domain = extract_registered_domain(
        "https://mail.google.com/mail/u/0/#inbox"
    )

    assert domain == "google.com"


def test_extract_booking_registered_domain():
    domain = extract_registered_domain(
        "https://booking.com/payments"
    )

    assert domain == "booking.com"


def test_extract_multi_part_domain():
    domain = extract_registered_domain(
        "https://www.example.co.uk/page"
    )

    assert domain == "example.co.uk"


def test_reputation_files_exist():
    assert (
        DATA_DIR / "phishing_urls.txt"
    ).exists()

    assert (
        DATA_DIR / "popular_domains.txt"
    ).exists()


def test_google_is_popular_or_unknown():
    result = check_reputation(
        "https://www.google.com/"
    )

    assert result["status"] in {
        "popular",
        "unknown",
    }

    if result["status"] == "popular":
        assert result["domain"] == "google.com"
        assert result["score"] == 0


def test_booking_is_popular_or_unknown():
    result = check_reputation(
        "https://booking.com/payments"
    )

    assert result["status"] in {
        "popular",
        "unknown",
    }

    if result["status"] == "popular":
        assert result["domain"] == "booking.com"


def test_malicious_feed_contains_urls():
    malicious_urls = load_malicious_urls()

    assert len(malicious_urls) > 0


def test_phishing_feed_url_is_malicious():
    malicious_urls = load_malicious_urls()

    first_url = next(iter(malicious_urls))

    result = check_reputation(first_url)

    assert result["status"] == "malicious"
    assert result["score"] == 100


def test_unknown_domain():
    result = check_reputation(
        "https://domain-that-is-not-in-the-lists-example.com"
    )

    assert result["status"] == "unknown"
    assert result["score"] is None


def test_typosquatting_domain_is_detected():
    result = check_reputation(
        "https://rnicrosoft.com"
    )

    assert result["status"] == "suspicious"
    assert result["score"] == 60
    assert result["domain"] == "rnicrosoft.com"
    assert result["similar_domain"] == "microsoft.com"

def test_malformed_urls_do_not_raise():
    for url in [
        "http://[invalid-ipv6/login",
        "http://example.com:99999/",
        "http://example.com:port/",
    ]:
        result = check_reputation(url)

        assert result["status"] in {
            "unknown",
            "suspicious",
            "popular",
        }
