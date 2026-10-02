from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit


CURRENT_DIRECTORY = Path(__file__).resolve().parent

if str(CURRENT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIRECTORY))

from train_combined_model import (
    prepare_dataset,
)


MODEL_PATH = (
    CURRENT_DIRECTORY
    / "char_combined_model.joblib"
)


TEST_URLS = {
    "Google": (
        "https://www.google.com/"
    ),
    "Booking": (
        "https://booking.com/payments"
    ),
    "Booking with query": (
        "https://www.booking.com/"
        "?chal_t=1790381880757"
        "&force_referer="
    ),
    "Microsoft": (
        "https://www.microsoft.com/"
    ),
    "Gmail": (
        "https://mail.google.com/"
        "mail/u/0/#inbox"
    ),
    "GitHub": (
        "https://github.com/"
    ),
    "Typosquatting Microsoft": (
        "https://rnicrosoft.com/"
    ),
    "Suspicious login": (
        "http://secure-login-verification-example.com/"
        "login"
    ),
    "Suspicious account": (
        "http://account-confirmation-example.com/"
        "verify/password"
    ),
}


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

    train_urls = urls.iloc[train_indices]
    test_urls = urls.iloc[test_indices]

    train_target = target.iloc[train_indices]
    test_target = target.iloc[test_indices]

    print("Training character-level model...")

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=250_000,
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

    print("Confusion matrix:")
    print(
        confusion_matrix(
            test_target,
            predictions,
        )
    )

    print("\nExternal URL evaluation:")
    print("-" * 70)

    external_urls = list(
        TEST_URLS.values()
    )

    external_matrix = vectorizer.transform(
        external_urls
    )

    external_probabilities = (
        model.predict_proba(
            external_matrix
        )[:, 1]
    )

    for (
        (name, url),
        probability,
    ) in zip(
        TEST_URLS.items(),
        external_probabilities,
    ):
        prediction = (
            "phishing"
            if probability >= 0.50
            else "legitimate"
        )

        print(f"\n{name}")
        print(f"URL: {url}")
        print(f"Prediction: {prediction}")
        print(
            "Phishing probability: "
            f"{probability * 100:.2f}%"
        )

    artifact = {
        "vectorizer": vectorizer,
        "model": model,
        "model_name": (
            "Character-level SGD"
        ),
    }

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()