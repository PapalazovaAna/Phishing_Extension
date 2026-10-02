from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit


CURRENT_DIRECTORY = Path(__file__).resolve().parent

if str(CURRENT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIRECTORY))

from legitphish_features import (
    extract_legitphish_features,
    feature_names,
)


DATASET_PATH = (
    CURRENT_DIRECTORY
    / "data"
    / "url_features_extracted1.csv"
)

MODEL_DIRECTORY = CURRENT_DIRECTORY


EXTERNAL_URLS = {
    "legitimate_google": (
        "https://www.google.com/"
    ),
    "legitimate_booking": (
        "https://booking.com/payments"
    ),
    "legitimate_booking_query": (
        "https://www.booking.com/"
        "?chal_t=1790381880757"
        "&force_referer="
    ),
    "legitimate_microsoft": (
        "https://www.microsoft.com/"
    ),
    "legitimate_gmail": (
        "https://mail.google.com/"
        "mail/u/0/#inbox"
    ),
    "legitimate_github": (
        "https://github.com/"
    ),
    "typosquatting_microsoft": (
        "https://rnicrosoft.com/"
    ),
    "suspicious_login": (
        "http://secure-login-verification-example.com/"
        "login"
    ),
    "suspicious_account": (
        "http://account-confirmation-example.com/"
        "verify/password"
    ),
}


def registered_domain(url: str) -> str:
    from urllib.parse import urlparse

    hostname = urlparse(url).hostname or ""
    parts = hostname.lower().split(".")

    if len(parts) >= 2:
        return ".".join(parts[-2:])

    return hostname.lower()


def prepare_dataset() -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    dataframe = pd.read_csv(DATASET_PATH)

    dataframe = dataframe[
        ["URL", "ClassLabel"]
    ].dropna()

    dataframe = dataframe.drop_duplicates(
        subset=["URL"]
    )

    dataframe["ClassLabel"] = (
        dataframe["ClassLabel"].astype(int)
    )

    feature_dataframe = pd.DataFrame(
        [
            extract_legitphish_features(url)
            for url in dataframe["URL"]
        ]
    )

    target = (
        dataframe["ClassLabel"] == 0
    ).astype(int)

    groups = dataframe["URL"].apply(
        registered_domain
    )

    return (
        feature_dataframe,
        target,
        groups,
    )


def create_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
            min_samples_leaf=2,
            max_features="sqrt",
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
            min_samples_leaf=2,
            max_features="sqrt",
        ),
        "HistGradientBoosting": (
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.08,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=42,
            )
        ),
    }


def evaluate_external_urls(
    model,
    model_name: str,
    feature_columns: list[str],
) -> None:
    print("\nExternal URL evaluation:")
    print("-" * 70)

    for name, url in EXTERNAL_URLS.items():
        features = extract_legitphish_features(url)

        row = pd.DataFrame(
            [[features[column] for column in feature_columns]],
            columns=feature_columns,
        )

        probability = model.predict_proba(row)[0][1]
        prediction = int(probability >= 0.5)

        label = (
            "phishing"
            if prediction == 1
            else "legitimate"
        )

        print(
            f"{name:<30}"
            f"{label:<12}"
            f"{probability * 100:>7.2f}%"
        )


def main() -> None:
    features, target, groups = prepare_dataset()

    feature_columns = feature_names()

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(
            features,
            target,
            groups=groups,
        )
    )

    x_train = features.iloc[train_indices]
    x_test = features.iloc[test_indices]

    y_train = target.iloc[train_indices]
    y_test = target.iloc[test_indices]

    print("Dataset shape:")
    print(features.shape)

    print(f"\nTraining records: {len(x_train)}")
    print(f"Test records: {len(x_test)}")

    print("\nClass distribution:")
    print(
        target.map(
            {
                0: "legitimate",
                1: "phishing",
            }
        ).value_counts()
    )

    models = create_models()
    model_results = []

    for model_name, model in models.items():
        print("\n" + "=" * 70)
        print(model_name)
        print("=" * 70)

        model.fit(x_train, y_train)

        predictions = model.predict(x_test)

        report = classification_report(
            y_test,
            predictions,
            target_names=[
                "legitimate",
                "phishing",
            ],
            digits=4,
        )

        print("\nClassification report:")
        print(report)

        print("Accuracy:")
        print(
            f"{accuracy_score(y_test, predictions):.4f}"
        )

        print("\nConfusion matrix:")
        print(
            confusion_matrix(
                y_test,
                predictions,
            )
        )

        phishing_report = classification_report(
            y_test,
            predictions,
            output_dict=True,
        )

        phishing_f1 = phishing_report[
            "1"
        ]["f1-score"]

        model_results.append(
            (
                phishing_f1,
                model_name,
                model,
            )
        )

        evaluate_external_urls(
            model=model,
            model_name=model_name,
            feature_columns=feature_columns,
        )

    model_results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    best_f1, best_name, best_model = (
        model_results[0]
    )

    artifact = {
        "model": best_model,
        "feature_names": feature_columns,
        "model_name": best_name,
        "phishing_f1": best_f1,
    }

    output_path = (
        MODEL_DIRECTORY
        / "legitphish_v2_model.joblib"
    )

    joblib.dump(artifact, output_path)

    print("\n" + "=" * 70)
    print("Best model")
    print("=" * 70)
    print(f"Model: {best_name}")
    print(f"Phishing F1: {best_f1:.4f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()