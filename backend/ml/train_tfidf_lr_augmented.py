from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from ml.benchmark_models import load_main_dataset


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "app" / "ml" / "data"

ORIGINAL_HOLDOUT_PATH = DATA_DIR / "external_evaluation_urls.csv"
INDEPENDENT_PATH = DATA_DIR / "independent_evaluation_urls.csv"

MODEL_PATH = BASE_DIR / "ml" / "tfidf_logistic_augmented.joblib"
PREDICTIONS_PATH = (
    BASE_DIR / "ml" / "tfidf_logistic_augmented_predictions.csv"
)

SEED = 42
MAX_TRAIN_PER_CLASS = 50_000


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


def split_by_domain(frame: pd.DataFrame):
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    train_indices, test_indices = next(
        splitter.split(
            frame["url"],
            frame["label"],
            groups=frame["domain"],
        )
    )

    return (
        frame.iloc[train_indices].copy(),
        frame.iloc[test_indices].copy(),
    )


def balanced_sample(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []

    for _, group in frame.groupby("label"):
        sample_size = min(
            len(group),
            MAX_TRAIN_PER_CLASS,
        )

        parts.append(
            group.sample(
                sample_size,
                random_state=SEED,
            )
        )

    return pd.concat(parts, ignore_index=True)


def print_metrics(name, labels, predictions):
    print(f"\n{name}")
    print("-" * 70)

    print(
        classification_report(
            labels,
            predictions,
            target_names=[
                "legitimate",
                "phishing",
            ],
            digits=4,
        )
    )


def main():
    print("Loading original datasets...")

    original = load_main_dataset()

    if ORIGINAL_HOLDOUT_PATH.exists():
        original_holdout = pd.read_csv(
            ORIGINAL_HOLDOUT_PATH
        )

        excluded_urls = set(
            original_holdout["url"].astype(str)
        )

        original = original[
            ~original["url"].astype(str).isin(
                excluded_urls
            )
        ].copy()

    curated_domains = {
        registered_domain(url)
        for url, _ in TEST_URLS
    }

    original["domain"] = original["url"].map(
        registered_domain
    )

    original = original[
        ~original["domain"].isin(curated_domains)
    ].copy()

    original_train, original_test = split_by_domain(
        original
    )

    original_train = balanced_sample(
        original_train
    )

    print(
        f"Original training records: "
        f"{len(original_train)}"
    )

    print(
        f"Original test records: "
        f"{len(original_test)}"
    )

    independent = pd.read_csv(
        INDEPENDENT_PATH
    )

    independent["url"] = (
        independent["url"]
        .astype(str)
        .str.strip()
    )

    independent["label"] = (
        independent["label"]
        .astype(int)
    )

    independent["domain"] = independent["url"].map(
        registered_domain
    )

    independent = independent[
        ~independent["domain"].isin(curated_domains)
    ].copy()

    independent_augmented, independent_final = (
        split_by_domain(independent)
    )

    print(
        "Independent augmentation records:",
        len(independent_augmented),
    )

    print(
        "Independent final records:",
        len(independent_final),
    )

    augmented_train = pd.concat(
        [
            original_train,
            independent_augmented,
        ],
        ignore_index=True,
    )

    augmented_train = augmented_train.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    print(
        "Total augmented training records:",
        len(augmented_train),
    )

    model = Pipeline([
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

    print("\nTraining augmented TF-IDF Logistic Regression...")

    model.fit(
        augmented_train["url"],
        augmented_train["label"],
    )

    # Original domain-aware test
    original_predictions = model.predict(
        original_test["url"]
    )

    print_metrics(
        "Original domain-aware test",
        original_test["label"],
        original_predictions,
    )

    # Independent final test
    independent_predictions = model.predict(
        independent_final["url"]
    )

    print_metrics(
        "Independent final test",
        independent_final["label"],
        independent_predictions,
    )

    # Curated URL test
    curated_urls = [
        url
        for url, _ in TEST_URLS
    ]

    curated_labels = [
        label
        for _, label in TEST_URLS
    ]

    curated_probabilities = model.predict_proba(
        curated_urls
    )[:, 1]

    curated_predictions = (
        curated_probabilities >= 0.5
    ).astype(int)

    print_metrics(
        "Curated URL test",
        curated_labels,
        curated_predictions,
    )

    print("\nCURATED URL RESULTS")
    print("=" * 80)

    for index, (url, expected) in enumerate(
        TEST_URLS
    ):
        probability = curated_probabilities[index]
        prediction = int(curated_predictions[index])

        expected_text = (
            "phishing"
            if expected == 1
            else "legitimate"
        )

        prediction_text = (
            "phishing"
            if prediction == 1
            else "legitimate"
        )

        print(f"\nURL: {url}")
        print(f"Expected: {expected_text}")
        print(f"Prediction: {prediction_text}")
        print(
            f"Phishing probability: "
            f"{probability:.2%}"
        )

    predictions_frame = independent_final[
        ["url", "label"]
    ].copy()

    predictions_frame["prediction"] = (
        independent_predictions
    )

    predictions_frame.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    joblib.dump(
        model,
        MODEL_PATH,
    )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )

    print(
        f"Predictions saved to: "
        f"{PREDICTIONS_PATH}"
    )


if __name__ == "__main__":
    main()