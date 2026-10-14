"""Detection of domains that imitate popular domains.

Four techniques are checked, from most to least specific:

1. Brand in subdomain   - paypal.com.account-check.xyz
2. Look-alike characters - rnicrosoft.com, micros0ft.com, xn--pple-43d.com
3. One-character edits   - gooogle.com, amazom.com, faecbook.com
4. Brand combined with phishing words - paypal-login.com, appleid-verify.net

The index is built once from the popular domain list, so each
lookup only needs a few dictionary operations.
"""

from dataclasses import dataclass


# Only the most popular domains are used as brands for the
# edit-distance check, less popular domains cause too many
# false positives (e.g. "hotel" vs "hotels").
BRAND_LIST_SIZE = 10_000

# Short labels have many legitimate one-edit neighbours
# (bing / king / ring), so they are skipped.
MIN_EDIT_DISTANCE_LABEL_LENGTH = 5

# Frequently impersonated brands for the combosquatting check.
# A curated list is used because generic popular labels such as
# "news" or "shop" appear inside many legitimate domains.
TARGETED_BRANDS = {
    "adobe": "adobe.com",
    "airbnb": "airbnb.com",
    "amazon": "amazon.com",
    "americanexpress": "americanexpress.com",
    "apple": "apple.com",
    "binance": "binance.com",
    "booking": "booking.com",
    "chase": "chase.com",
    "coinbase": "coinbase.com",
    "dhl": "dhl.com",
    "docusign": "docusign.com",
    "dropbox": "dropbox.com",
    "ebay": "ebay.com",
    "facebook": "facebook.com",
    "fedex": "fedex.com",
    "github": "github.com",
    "gmail": "gmail.com",
    "google": "google.com",
    "icloud": "icloud.com",
    "instagram": "instagram.com",
    "linkedin": "linkedin.com",
    "metamask": "metamask.io",
    "microsoft": "microsoft.com",
    "netflix": "netflix.com",
    "office365": "office.com",
    "outlook": "outlook.com",
    "paypal": "paypal.com",
    "roblox": "roblox.com",
    "spotify": "spotify.com",
    "steam": "steampowered.com",
    "telegram": "telegram.org",
    "twitter": "twitter.com",
    "usps": "usps.com",
    "wellsfargo": "wellsfargo.com",
    "whatsapp": "whatsapp.com",
    "yahoo": "yahoo.com",
}

PHISHING_KEYWORDS = (
    "account",
    "auth",
    "billing",
    "confirm",
    "help",
    "id",
    "login",
    "online",
    "pay",
    "payment",
    "recovery",
    "secure",
    "security",
    "service",
    "signin",
    "support",
    "update",
    "verify",
    "wallet",
)

# Character sequences that look alike, replaced by the
# character they imitate.
ASCII_LOOKALIKES = (
    ("rn", "m"),
    ("vv", "w"),
    ("0", "o"),
    ("1", "l"),
    ("3", "e"),
    ("5", "s"),
)

# Cyrillic and Greek letters that look like Latin letters,
# used in internationalized (xn--) homograph domains.
UNICODE_LOOKALIKES = str.maketrans(
    {
        "а": "a",
        "е": "e",
        "о": "o",
        "р": "p",
        "с": "c",
        "у": "y",
        "х": "x",
        "і": "i",
        "ј": "j",
        "ѕ": "s",
        "һ": "h",
        "ԁ": "d",
        "ӏ": "l",
        "α": "a",
        "ο": "o",
        "ν": "v",
        "ρ": "p",
    }
)


@dataclass(frozen=True)
class TyposquattingMatch:
    similar_domain: str
    technique: str
    description: str


def split_domain(domain: str) -> tuple[str, str]:
    parts = domain.split(".")

    if len(parts) < 2:
        return domain, ""

    return parts[0], ".".join(parts[1:])


def decode_punycode(label: str) -> str:
    if not label.startswith("xn--"):
        return label

    try:
        return label.encode("ascii").decode("idna")
    except UnicodeError:
        return label


def normalize_domain_label(label: str) -> str:
    label = decode_punycode(label.lower())
    label = label.translate(UNICODE_LOOKALIKES)

    for lookalike, original in ASCII_LOOKALIKES:
        label = label.replace(lookalike, original)

    return label


def single_deletions(label: str) -> set[str]:
    return {
        label[:index] + label[index + 1:]
        for index in range(len(label))
    }


def is_one_edit_apart(first: str, second: str) -> bool:
    """One insertion, deletion, substitution or swap of
    two neighbouring characters (Damerau-Levenshtein 1)."""

    if first == second:
        return False

    if abs(len(first) - len(second)) > 1:
        return False

    if len(first) == len(second):
        differences = [
            index
            for index, (a, b) in enumerate(zip(first, second))
            if a != b
        ]

        if len(differences) == 1:
            return True

        if len(differences) == 2:
            i, j = differences

            return (
                j == i + 1
                and first[i] == second[j]
                and first[j] == second[i]
            )

        return False

    shorter, longer = sorted(
        (first, second),
        key=len,
    )

    return shorter in single_deletions(longer)


class TyposquattingIndex:
    def __init__(self, ranked_popular_domains: list[str]):
        self.popular_domains = set(ranked_popular_domains)

        # (suffix, normalized label) -> most popular domain
        self.lookalike_index: dict[tuple[str, str], str] = {}

        for domain in ranked_popular_domains:
            label, suffix = split_domain(domain)

            self.lookalike_index.setdefault(
                (suffix, normalize_domain_label(label)),
                domain,
            )

        # (suffix, label with one character removed) -> brands.
        # Two labels one edit apart always share such a key
        # (symmetric delete), candidates are verified afterwards.
        self.edit_index: dict[tuple[str, str], list[str]] = {}

        for domain in ranked_popular_domains[:BRAND_LIST_SIZE]:
            label, suffix = split_domain(domain)

            if len(label) < MIN_EDIT_DISTANCE_LABEL_LENGTH:
                continue

            for key in {label} | single_deletions(label):
                self.edit_index.setdefault(
                    (suffix, key),
                    [],
                ).append(domain)

    def find_brand_in_subdomain(
        self,
        hostname: str,
        domain: str,
    ) -> TyposquattingMatch | None:
        subdomain = hostname.removesuffix(domain).rstrip(".")

        if not subdomain:
            return None

        for brand_domain in TARGETED_BRANDS.values():
            if (
                brand_domain != domain
                and f".{brand_domain}." in f".{subdomain}."
            ):
                return TyposquattingMatch(
                    similar_domain=brand_domain,
                    technique="brand_in_subdomain",
                    description=(
                        f"The address contains '{brand_domain}' "
                        f"as a subdomain, but the actual domain "
                        f"is '{domain}'."
                    ),
                )

        return None

    def find_lookalike(
        self,
        label: str,
        suffix: str,
        domain: str,
    ) -> TyposquattingMatch | None:
        similar_domain = self.lookalike_index.get(
            (suffix, normalize_domain_label(label))
        )

        if similar_domain and similar_domain != domain:
            return TyposquattingMatch(
                similar_domain=similar_domain,
                technique="lookalike_characters",
                description=(
                    f"The domain uses look-alike characters to "
                    f"imitate the popular domain "
                    f"'{similar_domain}'."
                ),
            )

        return None

    def find_one_edit_neighbour(
        self,
        label: str,
        suffix: str,
    ) -> TyposquattingMatch | None:
        if len(label) < MIN_EDIT_DISTANCE_LABEL_LENGTH - 1:
            return None

        for key in {label} | single_deletions(label):
            for brand in self.edit_index.get((suffix, key), []):
                brand_label, _ = split_domain(brand)

                if is_one_edit_apart(label, brand_label):
                    return TyposquattingMatch(
                        similar_domain=brand,
                        technique="one_character_edit",
                        description=(
                            f"The domain differs by one character "
                            f"from the popular domain '{brand}'."
                        ),
                    )

        return None

    def find_brand_with_keyword(
        self,
        label: str,
        domain: str,
    ) -> TyposquattingMatch | None:
        tokens = normalize_domain_label(label).split("-")

        has_keyword_token = any(
            token in PHISHING_KEYWORDS
            for token in tokens
        )

        for brand, brand_domain in TARGETED_BRANDS.items():
            if brand_domain == domain:
                continue

            for token in tokens:
                # paypal-login.com, but not apple-pie-recipes.com
                is_brand_token = token == brand and has_keyword_token

                # paypallogin.com, appleid.net
                is_joined_with_keyword = any(
                    token in (brand + keyword, keyword + brand)
                    for keyword in PHISHING_KEYWORDS
                )

                if is_brand_token or is_joined_with_keyword:
                    return TyposquattingMatch(
                        similar_domain=brand_domain,
                        technique="brand_with_keyword",
                        description=(
                            f"The domain combines the brand "
                            f"'{brand}' with other words, but is "
                            f"not the official domain "
                            f"'{brand_domain}'."
                        ),
                    )

        return None

    def find(
        self,
        domain: str,
        hostname: str | None = None,
    ) -> TyposquattingMatch | None:
        if hostname:
            match = self.find_brand_in_subdomain(hostname, domain)

            if match:
                return match

        if domain in self.popular_domains:
            return None

        label, suffix = split_domain(domain)

        return (
            self.find_lookalike(label, suffix, domain)
            or self.find_one_edit_neighbour(label, suffix)
            or self.find_brand_with_keyword(label, domain)
        )
