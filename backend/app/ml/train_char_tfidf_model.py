from pathlib import Path

import joblib
import pandas as pd
import tldextract

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_PATH = BASE_DIR / "char_tfidf_model.joblib"

PHIUSIIL_PATH = DATA_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
LEGITPHISH_PATH = DATA_DIR / "url_features_extracted1.csv"


EXTERNAL_URLS = {
    "Google": (
        "https://www.google.com/",
        0,
    ),
    "Gmail": (
        "https://mail.google.com/mail/u/0/#inbox",
        0,
    ),
    "Gmail with query": (
        "https://mail.google.com/mail/u/0/?tab=rm&ogbl#inbox",
        0,
    ),
    "Booking": (
        "https://booking.com/payments",
        0,
    ),
    "Booking with query": (
        "https://www.booking.com/"
        "?chal_t=1790763960679&force_referer=",
        0,
    ),
    "FINKI": (
        "https://finki.ukim.mk/",
        0,
    ),
    "FINKI undergraduate studies": (
        "https://finki.ukim.mk/"
        "studii-2/vidovi-studii/dodiplomski-studii/",
        0,
    ),
    "FINKI masters quotas": (
        "https://finki.ukim.mk/"
        "upisi/magisterski-studii/kvoti/",
        0,
    ),
    "Microsoft": (
        "https://www.microsoft.com/",
        0,
    ),
    "GitHub": (
        "https://github.com/",
        0,
    ),
    "Python documentation": (
        "https://docs.python.org/3/library/urllib.parse.html",
        0,
    ),
    "Typosquatting Microsoft": (
        "https://rnicrosoft.com/",
        1,
    ),
    "Suspicious login": (
        "http://secure-login-verification.example/login",
        1,
    ),
    "Suspicious account": (
        "http://account-confirmation.example/verify/password",
        1,
    ),
    "Suspicious payment": (
        "http://paypal-secure.example/verify/account",
        1,
    ),
    "IP address URL": (
        "http://192.0.2.10/login",
        1,
    ),
    "Encoded phishing URL": (
        "http://example.invalid/%6cogin/%76erify/password",
        1,
    ),
}


def get_registered_domain(url: str) -> str:
    extracted = tldextract.extract(url)

    registered_domain = (
        extracted.top_domain_under_public_suffix
    )

    if registered_domain:
        return registered_domain.lower()

    return extracted.domain.lower()


def load_phiusiil() -> pd.DataFrame:
    dataframe = pd.read_csv(PHIUSIIL_PATH)

    dataframe = dataframe.rename(
        columns={
            "URL": "url",
            "label": "label",
        }
    )

    dataframe = dataframe[
        ["url", "label"]
    ].copy()

    dataframe["label"] = pd.to_numeric(
        dataframe["label"],
        errors="coerce",
    )

    # Raw PhiUSIIL encoding:
    # 0 = phishing
    # 1 = legitimate
    #
    # Common project encoding:
    # 0 = legitimate
    # 1 = phishing
    dataframe["label"] = 1 - dataframe["label"]

    dataframe["source"] = "PhiUSIIL"

    return dataframe


def load_legitphish() -> pd.DataFrame:
    dataframe = pd.read_csv(LEGITPHISH_PATH)

    dataframe = dataframe.rename(
        columns={
            "URL": "url",
            "ClassLabel": "label",
        }
    )

    dataframe = dataframe[
        ["url", "label"]
    ].copy()

    dataframe["label"] = pd.to_numeric(
        dataframe["label"],
        errors="coerce",
    )

    # Raw LegitPhish encoding:
    # 0 = phishing
    # 1 = legitimate
    dataframe["label"] = 1 - dataframe["label"]

    dataframe["source"] = "LegitPhish"

    return dataframe
    

def load_dataset() -> pd.DataFrame:
    phiusiil = load_phiusiil()
    legitphish = load_legitphish()

    dataframe = pd.concat(
        [phiusiil, legitphish],
        ignore_index=True,
    )

    dataframe["url"] = (
        dataframe["url"]
        .astype(str)
        .str.strip()
    )

    dataframe["label"] = pd.to_numeric(
        dataframe["label"],
        errors="coerce",
    )

    dataframe = dataframe.dropna(
        subset=["url", "label"]
    )

    dataframe["label"] = (
        dataframe["label"]
        .astype(int)
    )

    dataframe = dataframe[
        dataframe["label"].isin([0, 1])
    ]

    dataframe = dataframe[
        dataframe["url"].str.len() > 0
    ]

    # Remove exact duplicates.
    dataframe = dataframe.drop_duplicates(
        subset=["url", "label"]
    )

    # Remove URLs that appear with conflicting labels.
    label_counts = (
        dataframe.groupby("url")["label"]
        .nunique()
    )

    conflicting_urls = label_counts[
        label_counts > 1
    ].index

    dataframe = dataframe[
        ~dataframe["url"].isin(conflicting_urls)
    ]

    dataframe["domain"] = dataframe["url"].apply(
        get_registered_domain
    )

    return dataframe.reset_index(drop=True)


def build_model() -> Pipeline:
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=250_000,
        sublinear_tf=True,
        lowercase=True,
    )

    classifier = LogisticRegression(
        C=2.0,
        class_weight="balanced",
        max_iter=300,
        solver="saga",
        n_jobs=-1,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ]
    )


def evaluate_external_urls(model: Pipeline) -> None:
    print()
    print("=" * 70)
    print("EXTERNAL URL EVALUATION")
    print("=" * 70)

    legitimate_total = 0
    legitimate_false_positives = 0

    phishing_total = 0
    phishing_true_positives = 0

    for name, (url, expected_label) in EXTERNAL_URLS.items():
        probabilities = model.predict_proba([url])[0]
        phishing_probability = float(
            probabilities[1]
        )

        prediction = int(
            phishing_probability >= 0.50
        )

        prediction_name = (
            "phishing"
            if prediction == 1
            else "legitimate"
        )

        expected_name = (
            "phishing"
            if expected_label == 1
            else "legitimate"
        )

        print()
        print(name)
        print(f"URL: {url}")
        print(f"Expected: {expected_name}")
        print(f"Prediction: {prediction_name}")
        print(
            "Phishing probability: "
            f"{phishing_probability * 100:.2f}%"
        )

        if expected_label == 0:
            legitimate_total += 1

            if prediction == 1:
                legitimate_false_positives += 1

        else:
            phishing_total += 1

            if prediction == 1:
                phishing_true_positives += 1

    print()
    print("=" * 70)
    print("EXTERNAL SUMMARY")
    print("=" * 70)
    print(
        "Legitimate false positives: "
        f"{legitimate_false_positives}/"
        f"{legitimate_total}"
    )
    print(
        "Phishing true positives: "
        f"{phishing_true_positives}/"
        f"{phishing_total}"
    )


def main() -> None:
    dataframe = load_dataset()

    print("Dataset shape:")
    print(dataframe.shape)

    print()
    print("Source distribution:")
    print(dataframe["source"].value_counts())

    print()
    print("Class distribution:")
    print(dataframe["label"].value_counts())

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(
            dataframe["url"],
            dataframe["label"],
            groups=dataframe["domain"],
        )
    )

    train_data = dataframe.iloc[
        train_indices
    ].copy()

    test_data = dataframe.iloc[
        test_indices
    ].copy()

    print()
    print(f"Training records: {len(train_data)}")
    print(f"Test records: {len(test_data)}")
    print(
        "Training domains: "
        f"{train_data['domain'].nunique()}"
    )
    print(
        "Test domains: "
        f"{test_data['domain'].nunique()}"
    )

    model = build_model()

    print()
    print("Training character TF-IDF model...")

    model.fit(
        train_data["url"],
        train_data["label"],
    )

    predictions = model.predict(
        test_data["url"]
    )

    print()
    print("Classification report:")
    print(
        classification_report(
            test_data["label"],
            predictions,
            target_names=[
                "legitimate",
                "phishing",
            ],
            digits=4,
        )
    )

    print("Accuracy:")
    print(
        f"{accuracy_score(test_data['label'], predictions):.4f}"
    )

    print()
    print("Confusion matrix:")
    print(
        confusion_matrix(
            test_data["label"],
            predictions,
        )
    )

    joblib.dump(model, MODEL_PATH)

    print()
    print(f"Model saved to: {MODEL_PATH}")

    evaluate_external_urls(model)


if __name__ == "__main__":
    main()