from functools import lru_cache


def normalize_domain_label(label: str) -> str:
    return (
        label.lower()
        .replace("rn", "m")
        .replace("0", "o")
        .replace("1", "l")
        .replace("3", "e")
        .replace("5", "s")
    )


def split_domain(domain: str) -> tuple[str, str]:
    parts = domain.split(".")

    if len(parts) < 2:
        return domain, ""

    return parts[0], ".".join(parts[1:])


@lru_cache(maxsize=1)
def build_domain_index(
    popular_domains: tuple[str, ...]
) -> dict[tuple[str, str], str]:
    index: dict[tuple[str, str], str] = {}

    for domain in popular_domains:
        label, suffix = split_domain(domain)

        normalized_label = normalize_domain_label(label)

        index.setdefault(
            (suffix, normalized_label),
            domain,
        )

    return index


def find_similar_popular_domain(
    domain: str,
    popular_domains: set[str],
) -> str | None:
    if domain in popular_domains:
        return None

    label, suffix = split_domain(domain)

    normalized_label = normalize_domain_label(label)

    popular_domains_tuple = tuple(
        sorted(popular_domains)
    )

    domain_index = build_domain_index(
        popular_domains_tuple
    )

    similar_domain = domain_index.get(
        (suffix, normalized_label)
    )

    if similar_domain and similar_domain != domain:
        return similar_domain

    return None