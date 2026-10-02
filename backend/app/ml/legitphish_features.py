import math
import re
from urllib.parse import urlparse


SUSPICIOUS_WORDS = {
    "login",
    "signin",
    "verify",
    "secure",
    "account",
    "password",
    "confirm",
    "update",
    "payment",
    "wallet",
    "support",
}


SUSPICIOUS_EXTENSIONS = {
    ".exe",
    ".scr",
    ".zip",
    ".rar",
    ".js",
    ".html",
    ".htm",
    ".php",
    ".asp",
    ".aspx",
}


def _entropy(value: str) -> float:
    if not value:
        return 0.0

    frequencies = {
        character: value.count(character)
        for character in set(value)
    }

    length = len(value)

    return -sum(
        (frequency / length)
        * math.log2(frequency / length)
        for frequency in frequencies.values()
    )


def _is_ip_address(hostname: str) -> bool:
    parts = hostname.split(".")

    return len(parts) == 4 and all(
        part.isdigit() and 0 <= int(part) <= 255
        for part in parts
    )


def _registered_domain_parts(hostname: str) -> tuple[str, str]:
    parts = hostname.lower().split(".")

    if len(parts) >= 2:
        return parts[-2], parts[-1]

    if len(parts) == 1:
        return parts[0], ""

    return "", ""


def extract_legitphish_features(
    url: str,
) -> dict[str, float]:
    parsed_url = urlparse(url)

    hostname = parsed_url.hostname or ""
    hostname = hostname.lower()

    path = parsed_url.path or ""
    query = parsed_url.query or ""
    lower_url = url.lower()

    domain_name, tld = _registered_domain_parts(hostname)

    tokens = [
        token
        for token in re.split(r"[^a-zA-Z0-9]+", url)
        if token
    ]

    query_param_count = (
        len(query.split("&"))
        if query
        else 0
    )

    suspicious_file_extension = float(
        any(
            path.lower().endswith(extension)
            for extension in SUSPICIOUS_EXTENSIONS
        )
    )

    suspicious_word_count = sum(
        word in lower_url
        for word in SUSPICIOUS_WORDS
    )

    numeric_characters = sum(
        character.isdigit()
        for character in url
    )

    percentage_numeric_chars = (
        numeric_characters / len(url) * 100
        if url
        else 0
    )

    return {
        "url_length": float(len(url)),
        "has_ip_address": float(
            _is_ip_address(hostname)
        ),
        "dot_count": float(url.count(".")),
        "https_flag": float(
            parsed_url.scheme.lower() == "https"
        ),
        "url_entropy": _entropy(url),
        "token_count": float(len(tokens)),
        "subdomain_count": float(
            max(len(hostname.split(".")) - 2, 0)
        ),
        "query_param_count": float(
            query_param_count
        ),
        "tld_length": float(len(tld)),
        "path_length": float(len(path)),
        "has_hyphen_in_domain": float(
            "-" in domain_name
        ),
        "number_of_digits": float(
            numeric_characters
        ),
        "suspicious_file_extension": (
            suspicious_file_extension
        ),
        "domain_name_length": float(
            len(domain_name)
        ),
        "percentage_numeric_chars": float(
            percentage_numeric_chars
        ),
    }


def feature_names() -> list[str]:
    return list(
        extract_legitphish_features(
            "https://example.com"
        ).keys()
    )