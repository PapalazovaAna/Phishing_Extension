from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit


CURRENT_DIRECTORY = Path(__file__).resolve().parent

if str(CURRENT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIRECTORY))

from train_combined_model import (
    prepare_dataset,
    get_registered_domain,
)


POPULAR_DOMAINS_PATH = (
    CURRENT_DIRECTORY.parent
    / "reputation_data"
    / "popular_domains.txt"
)

MODEL_PATH = (
    CURRENT_DIRECTORY
    / "char_hard_negative_model.joblib"
)


EXTERNAL_URLS = {
    "Google": "https://www.google.com/",
    "Booking": "https://booking.com/payments",
    "Booking query": (
        "https://www.booking.com/"
        "?chal_t=1790381880757&force_referer="
    ),
    "Microsoft": "https://www.microsoft.com/",
    "Gmail": (
        "https://mail.google.com/mail/u/0/#inbox"
    ),
    "GitHub": "https://github.com/",
    "Typosquatting Microsoft": (
        "https://rnicrosoft.com/"
    ),
    "Suspicious login": (
        "http://secure-login-verification-example.com/login"
    ),
    "Suspicious account": (
        "http://account-confirmation-example.com/"
        "verify/password"
    ),
}


def read_popular_domains() -> list[str]:
    domains = []

    with open(
        POPULAR_DOMAINS_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            value = line.strip()

            if not value:
                continue

            if value.startswith("#"):
                continue

            if "," in value:
                value = value.split(",")[-1].strip()

            value = value.lower()

            if "." in value:
                domains.append(value)

    return list(dict.fromkeys(domains))


def main() -> None:
    dataframe = prepare_dataset()

    urls = dataframe["URL"]
    target = dataframe["target"]
    groups = dataframe["registered_domain"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(
            urls,
            target,
            groups=groups,
        )
    )

    train_urls = urls.iloc[train_indices].tolist()
    train_target = target.iloc[train_indices].tolist()

    test_urls = urls.iloc[test_indices].tolist()
    test_target = target.iloc[test_indices].tolist()

    excluded_domains = {
        get_registered_domain(url)
        for url in EXTERNAL_URLS.values()
    }

    popular_domains = read_popular_domains()

    popular_domains = [
        domain
        for domain in popular_domains
        if get_registered_domain(
            f"https://{domain}/"
        ) not in excluded_domains
    ]

    hard_negative_count = min(
        50_000,
        len(popular_domains),
    )

    popular_domains = popular_domains[
        :hard_negative_count
    ]

    hard_negative_urls = [
        f"https://{domain}/"
        for domain in popular_domains
    ]

    # Hard negatives are legitimate examples.
    train_urls.extend(hard_negative_urls)
    train_target.extend(
        [0] * len(hard_negative_urls)
    )

    print(
        f"Original training records: "
        f"{len(train_indices)}"
    )

    print(
        f"Added legitimate hard negatives: "
        f"{len(hard_negative_urls)}"
    )

    print(
        f"Final training records: "
        f"{len(train_urls)}"
    )

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=300_000,
        sublinear_tf=True,
    )

    train_matrix = vectorizer.fit_transform(
        train_urls
    )

    test_matrix = vectorizer.transform(
        test_urls
    )

    model = SGDClassifier(
        loss="log_loss",
        alpha=0.00001,
        max_iter=50,
        tol=0.0001,
        class_weight="balanced",
        random_state=42,
        early_stopping=True,
        validation_fraction=0.10,
        n_iter_no_change=5,
    )

    model.fit(
        train_matrix,
        train_target,
    )

    predictions = model.predict(
        test_matrix
    )

    print("\nClassification report:")
    print(
        classification_report(
            test_target,
            predictions,
            target_names=[
                "legitimate",
                "phishing",
            ],
            digits=4,
        )
    )

    print("\nExternal URL evaluation:")

    external_matrix = vectorizer.transform(
        list(EXTERNAL_URLS.values())
    )

    probabilities = model.predict_proba(
        external_matrix
    )[:, 1]

    for (
        (name, url),
        probability,
    ) in zip(
        EXTERNAL_URLS.items(),
        probabilities,
    ):
        classification = (
            "phishing"
            if probability >= 0.50
            else "legitimate"
        )

        print(f"\n{name}")
        print(f"URL: {url}")
        print(f"Prediction: {classification}")
        print(
            "Phishing probability: "
            f"{probability * 100:.2f}%"
        )

    artifact = {
        "vectorizer": vectorizer,
        "model": model,
        "model_name": (
            "Character model with hard negatives"
        ),
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()