import pytest

from app.reputation import check_reputation
from app.typosquatting import (
    TyposquattingIndex,
    is_one_edit_apart,
    normalize_domain_label,
)


POPULAR_DOMAINS = [
    "google.com",
    "microsoft.com",
    "paypal.com",
    "apple.com",
    "amazon.com",
    "facebook.com",
    "answers.com",
]


@pytest.fixture(scope="module")
def index():
    return TyposquattingIndex(POPULAR_DOMAINS)


@pytest.mark.parametrize(
    "first, second, expected",
    [
        ("google", "gooogle", True),
        ("amazon", "amazom", True),
        ("facebook", "faecbook", True),
        ("paypal", "paypl", True),
        ("google", "google", False),
        ("google", "goggle1", False),
        ("xab", "abx", False),
    ],
)
def test_is_one_edit_apart(first, second, expected):
    assert is_one_edit_apart(first, second) is expected


def test_normalize_replaces_lookalike_characters():
    assert normalize_domain_label("rnicros0ft") == "microsoft"
    assert normalize_domain_label("xn--pple-43d") == "apple"


@pytest.mark.parametrize(
    "domain, hostname, similar_domain, technique",
    [
        (
            "rnicrosoft.com",
            None,
            "microsoft.com",
            "lookalike_characters",
        ),
        (
            "xn--pple-43d.com",
            None,
            "apple.com",
            "lookalike_characters",
        ),
        (
            "amazom.com",
            None,
            "amazon.com",
            "one_character_edit",
        ),
        (
            "paypal-login.com",
            None,
            "paypal.com",
            "brand_with_keyword",
        ),
        (
            "appleid.net",
            None,
            "apple.com",
            "brand_with_keyword",
        ),
        (
            "account-check.xyz",
            "paypal.com.account-check.xyz",
            "paypal.com",
            "brand_in_subdomain",
        ),
    ],
)
def test_detects_typosquatting(
    index,
    domain,
    hostname,
    similar_domain,
    technique,
):
    match = index.find(domain, hostname=hostname)

    assert match is not None
    assert match.similar_domain == similar_domain
    assert match.technique == technique


@pytest.mark.parametrize(
    "domain, hostname",
    [
        ("google.com", "mail.google.com"),
        ("paypal.com", "www.paypal.com"),
        ("apple-pie-recipes.com", None),
        ("pineapple-shop.com", None),
        ("example.com", None),
    ],
)
def test_does_not_flag_legitimate_domains(index, domain, hostname):
    assert index.find(domain, hostname=hostname) is None


def test_reputation_reports_typosquatting_technique():
    result = check_reputation("https://paypal-secure-login.com/")

    assert result["status"] == "suspicious"
    assert result["similar_domain"] == "paypal.com"
    assert result["technique"] == "brand_with_keyword"
