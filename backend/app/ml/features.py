import math
import re
from urllib.parse import urlparse


SUSPICIOUS_WORDS = {
    "login",
    "verify",
    "secure",
    "account",
    "password",
    "confirm",
    "update",
}


def _is_ip_address(hostname: str) -> bool:
    parts = hostname.split(".")

    return len(parts) == 4 and all(
        part.isdigit() and 0 <= int(part) <= 255
        for part in parts
    )


def _entropy(value: str) -> float:
    if not value:
        return 0.0

    length = len(value)
    frequencies = {
        character: value.count(character)
        for character in set(value)
    }

    return -sum(
        (frequency / length) * math.log2(frequency / length)
        for frequency in frequencies.values()
    )


def extract_url_features(url: str) -> dict[str, float]:
    parsed_url = urlparse(url)
    hostname = parsed_url.hostname or ""
    path = parsed_url.path or ""
    lower_url = url.lower()

    return {
        "url_length": float(len(url)),
        "hostname_length": float(len(hostname)),
        "path_length": float(len(path)),
        "number_of_dots": float(url.count(".")),
        "number_of_hyphens": float(url.count("-")),
        "number_of_underscores": float(url.count("_")),
        "number_of_digits": float(
            sum(character.isdigit() for character in url)
        ),
        "number_of_subdomains": float(
            max(len(hostname.split(".")) - 2, 0)
        ),
        "number_of_parameters": float(
            len(parsed_url.query.split("&"))
            if parsed_url.query
            else 0
        ),
        "has_ip_address": float(_is_ip_address(hostname)),
        "has_https": float(parsed_url.scheme == "https"),
        "has_at_symbol": float("@" in url),
        "has_punycode": float("xn--" in hostname.lower()),
        "has_encoded_characters": float("%" in url),
        "suspicious_word_count": float(
            sum(word in lower_url for word in SUSPICIOUS_WORDS)
        ),
        "url_entropy": _entropy(url),
        "special_character_count": float(
            len(re.findall(r"[^a-zA-Z0-9]", url))
        ),
    }


def feature_names() -> list[str]:
    return list(
        extract_url_features("https://example.com").keys()
    )