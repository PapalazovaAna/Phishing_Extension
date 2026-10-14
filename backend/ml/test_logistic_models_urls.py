from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.ml.features import extract_url_features, feature_names
from ml.benchmark_models import load_main_dataset


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "app" / "ml" / "data"
ORIGINAL_HOLDOUT_PATH = DATA_DIR / "external_evaluation_urls.csv"


TEST_URLS = [
    ("https://www.google.com/", 0),
    ("https://mail.google.com/mail/u/0/#inbox", 0),
    ("https://finki.ukim.mk/", 0),
    (
        "https://finki.ukim.mk/studii-2/vidovi-studii/"
        "dodiplomski-studii/",
        0,
    ),
    ("https://www.microsoft.com/", 0),
    ("https://github.com/", 0),
    (
        "https://docs.python.org/3/library/urllib.parse.html",
        0,
    ),
    ("https://www.booking.com/", 0),
    (
        "http://secure-login-verification.example/login",
        1,
    ),
    (
        "http://account-confirmation.example/verify/password",
        1,
    ),
    (
        "http://paypal-secure.example/verify/account",
        1,
    ),
    (
        "http://192.0.2.10/login",
        1,
    ),
]


def registered_domain(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower().strip(".")
    parts = hostname.split(".")

    return ".".join(parts[-2:]) if len(parts) >= 2 else hostname


def main():
    frame = load_main_dataset()

    if ORIGINAL_HOLDOUT_PATH.exists():
        holdout = pd.read_csv(ORIGINAL_HOLDOUT_PATH)

        frame = frame[
            ~frame["url"].isin(set(holdout["url"]))
        ].copy()

    test_domains = {
        registered_domain(url)
        for url, _ in TEST_URLS
    }

    frame = frame[
        ~frame["url"].map(registered_domain).isin(test_domains)
    ].copy()

    print(f"Training records: {len(frame)}")
    print(f"Excluded test domains: {len(test_domains)}")

    test_urls = [url for url, _ in TEST_URLS]
    expected_labels = [label for _, label in TEST_URLS]

    # ---------------------------------------------------------
    # Model 1: Character TF-IDF + Logistic Regression
    # ---------------------------------------------------------

    tfidf_model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                analyzer="char",
                ngram_range=(3, 5),
                min_df=2,
                max_features=150_000,
                sublinear_tf=True,
            ),
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ])

    print("\nTraining Character TF-IDF + Logistic Regression...")
    tfidf_model.fit(frame["url"], frame["label"])

    tfidf_probabilities = (
        tfidf_model.predict_proba(test_urls)[:, 1]
    )

    # ---------------------------------------------------------
    # Model 2: Engineered URL Features + Logistic Regression
    # ---------------------------------------------------------

    training_features = pd.DataFrame(
        frame["url"].map(extract_url_features).tolist(),
        columns=feature_names(),
    )

    test_features = pd.DataFrame(
        [extract_url_features(url) for url in test_urls],
        columns=feature_names(),
    )

    tabular_model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
            ),
        ),
    ])

    print("Training engineered-features Logistic Regression...")
    tabular_model.fit(
        training_features,
        frame["label"],
    )

    tabular_probabilities = (
        tabular_model.predict_proba(test_features)[:, 1]
    )

    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    print("\nLOGISTIC REGRESSION URL TEST RESULTS")
    print("=" * 80)

    for index, (url, expected) in enumerate(TEST_URLS):
        expected_text = (
            "phishing"
            if expected == 1
            else "legitimate"
        )

        tfidf_probability = tfidf_probabilities[index]
        tabular_probability = tabular_probabilities[index]

        tfidf_prediction = (
            "phishing"
            if tfidf_probability >= 0.5
            else "legitimate"
        )

        tabular_prediction = (
            "phishing"
            if tabular_probability >= 0.5
            else "legitimate"
        )

        print(f"\nURL: {url}")
        print(f"Expected: {expected_text}")

        print(
            "TF-IDF Logistic Regression: "
            f"{tfidf_prediction} "
            f"({tfidf_probability:.2%})"
        )

        print(
            "Tabular Logistic Regression: "
            f"{tabular_prediction} "
            f"({tabular_probability:.2%})"
        )


if __name__ == "__main__":
    main()