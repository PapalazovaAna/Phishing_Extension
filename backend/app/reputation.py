from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse

import tldextract

from app.typosquatting import TyposquattingIndex


DATA_DIR = Path(__file__).parent / "reputation_data"

PHISHING_URLS_FILE = DATA_DIR / "phishing_urls.txt"
POPULAR_DOMAINS_FILE = DATA_DIR / "popular_domains.txt"


domain_extractor = tldextract.TLDExtract(
    include_psl_private_domains=True
)


def extract_hostname(url: str) -> str | None:
    try:
        hostname = urlparse(url).hostname
    except ValueError:
        return None

    if not hostname:
        return None

    return hostname.lower().strip(".")


def extract_registered_domain(url: str) -> str | None:
    try:
        hostname = extract_hostname(url)

        if not hostname:
            return None

        extracted = domain_extractor(hostname)

        if not extracted.domain or not extracted.suffix:
            return None

        return extracted.top_domain_under_public_suffix.lower()

    except ValueError:
        return None


def normalize_url(url: str) -> str:
    fallback = url.strip().lower().rstrip("/")

    try:
        parsed_url = urlparse(url.strip())

        if not parsed_url.scheme or not parsed_url.netloc:
            return fallback

        hostname = (parsed_url.hostname or "").lower()

        port = ""
        if parsed_url.port:
            port = f":{parsed_url.port}"

    # Malformed URLs, e.g. an invalid IPv6 host or port.
    except ValueError:
        return fallback

    path = parsed_url.path.rstrip("/")

    normalized = (
        f"{parsed_url.scheme.lower()}://"
        f"{hostname}{port}{path}"
    )

    if parsed_url.query:
        normalized += f"?{parsed_url.query}"

    return normalized


def _read_lines(file_path: Path) -> set[str]:
    if not file_path.exists():
        return set()

    return {
        line.strip().lower()
        for line in file_path.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()
        if line.strip()
    }


@lru_cache(maxsize=1)
def load_ranked_popular_domains() -> list[str]:
    # The file is in Tranco rank order, most popular first.
    if not POPULAR_DOMAINS_FILE.exists():
        return []

    return [
        line.strip().lower()
        for line in POPULAR_DOMAINS_FILE.read_text(
            encoding="utf-8",
            errors="ignore",
        ).splitlines()
        if line.strip()
    ]


@lru_cache(maxsize=1)
def load_popular_domains() -> set[str]:
    return set(load_ranked_popular_domains())


@lru_cache(maxsize=1)
def load_typosquatting_index() -> TyposquattingIndex:
    return TyposquattingIndex(load_ranked_popular_domains())


def load_reputation_data() -> None:
    """Load all lists up front so the first request is not slow."""

    load_malicious_urls()
    load_popular_domains()
    load_typosquatting_index()


@lru_cache(maxsize=1)
def load_malicious_urls() -> set[str]:
    phishing_urls = _read_lines(PHISHING_URLS_FILE)

    return {
        normalize_url(url)
        for url in phishing_urls
    }


def check_reputation(url: str) -> dict:
    domain = extract_registered_domain(url)
    normalized_url = normalize_url(url)

    if not domain:
        return {
            "status": "unknown",
            "score": None,
            "source": "local reputation files",
            "domain": None,
            "indicator": "The domain could not be extracted.",
        }

    malicious_urls = load_malicious_urls()
    popular_domains = load_popular_domains()

    if normalized_url in malicious_urls:
        return {
            "status": "malicious",
            "score": 100,
            "source": "OpenPhish feed",
            "domain": domain,
            "indicator": (
                "The exact URL is present in the phishing URL list."
            ),
        }

    typosquatting_match = load_typosquatting_index().find(
        domain,
        hostname=extract_hostname(url),
    )

    if typosquatting_match:
        return {
            "status": "suspicious",
            "score": 60,
            "source": "Tranco similarity analysis",
            "domain": domain,
            "similar_domain": typosquatting_match.similar_domain,
            "technique": typosquatting_match.technique,
            "indicator": typosquatting_match.description,
        }

    if domain in popular_domains:
        return {
            "status": "popular",
            "score": 0,
            "source": "Tranco top-domain list",
            "domain": domain,
            "indicator": (
                "The registered domain appears in the "
                "Tranco popularity list."
            ),
        }

    return {
        "status": "unknown",
        "score": None,
        "source": "OpenPhish and Tranco lists",
        "domain": domain,
        "indicator": (
            "The URL is not present in the phishing list and "
            "the domain is not in the popularity list."
        ),
    }