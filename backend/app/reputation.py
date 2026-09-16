from urllib.parse import urlparse


TRUSTED_DOMAINS = {
    "example.com",
    "github.com",
    "google.com",
    "microsoft.com",
}


MALICIOUS_DOMAINS = {
    "malicious.test",
    "phishing.test",
}


def belongs_to_domain(hostname: str, domain: str) -> bool:
    return (
        hostname == domain
        or hostname.endswith("." + domain)
    )


def check_reputation(url: str) -> dict:
    hostname = (
        urlparse(url).hostname
        or ""
    ).lower().strip(".")

    if any(
        belongs_to_domain(hostname, domain)
        for domain in MALICIOUS_DOMAINS
    ):
        return {
            "status": "malicious",
            "score": 100,
            "source": "local blacklist",
            "indicator": (
                "The domain is present in the "
                "local malicious-domain list."
            ),
        }

    if any(
        belongs_to_domain(hostname, domain)
        for domain in TRUSTED_DOMAINS
    ):
        return {
            "status": "trusted",
            "score": 0,
            "source": "local whitelist",
            "indicator": (
                "The domain is present in the "
                "local trusted-domain list."
            ),
        }

    return {
        "status": "unknown",
        "score": None,
        "source": None,
        "indicator": None,
    }