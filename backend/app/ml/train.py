from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix
)
from sklearn.model_selection import StratifiedGroupKFold

from features import (
    extract_url_features,
    feature_names
)


DATASET_PATH = Path(
    "app/ml/data/PhiUSIIL_Phishing_URL_Dataset.csv"
)

MODEL_PATH = Path(
    "app/ml/url_model.joblib"
)


def find_column(
    columns: list[str],
    expected_name: str
) -> str:
    matching_columns = [
        column
        for column in columns
        if column.lower() == expected_name.lower()
    ]

    if not matching_columns:
        raise ValueError(
            f"Could not find column '{expected_name}'. "
            f"Available columns: {columns}"
        )

    return matching_columns[0]


def convert_labels(
    labels: pd.Series
) -> pd.Series:
    numeric_labels = pd.to_numeric(
        labels,
        errors="coerce"
    )

    if numeric_labels.notna().all():
        # In PhiUSIIL:
        # 0 = phishing
        # 1 = legitimate
        return (numeric_labels == 0).astype(int)

    normalized_labels = (
        labels.astype(str)
        .str.lower()
        .str.strip()
    )

    phishing_values = {
        "phishing",
        "malicious",
        "true",
        "1"
    }

    legitimate_values = {
        "legitimate",
        "benign",
        "safe",
        "false",
        "0"
    }

    unknown_values = set(normalized_labels) - (
        phishing_values
        | legitimate_values
    )

    if unknown_values:
        raise ValueError(
            "Unknown label values: "
            + ", ".join(sorted(unknown_values))
        )

    return normalized_labels.isin(
        phishing_values
    ).astype(int)


def domain_from_url(url: str) -> str:
    hostname = (
        urlparse(url).hostname
        or ""
    ).lower()

    parts = hostname.split(".")

    if len(parts) >= 2:
        return ".".join(parts[-2:])

    return hostname


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found at: {DATASET_PATH}"
        )

    dataset = pd.read_csv(DATASET_PATH)

    columns = list(dataset.columns)

    url_column = find_column(
        columns,
        "url"
    )

    label_column = find_column(
        columns,
        "label"
    )

    dataset = dataset[
        [url_column, label_column]
    ].copy()

    dataset.columns = [
        "url",
        "label"
    ]

    dataset = dataset.dropna(
        subset=[
            "url",
            "label"
        ]
    )

    dataset["url"] = (
        dataset["url"]
        .astype(str)
        .str.strip()
    )

    dataset = dataset[
        dataset["url"] != ""
    ]

    dataset = dataset.drop_duplicates(
        subset=["url"]
    )

    dataset["domain_group"] = dataset["url"].map(
        domain_from_url
    )

    features = pd.DataFrame(
        dataset["url"]
        .map(extract_url_features)
        .tolist(),
        columns=feature_names()
    )

    features = features.drop(
        columns=["has_https"]
    )

    labels = convert_labels(
        dataset["label"]
    )

    groups = dataset["domain_group"]

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    train_indices, test_indices = next(
        splitter.split(
            features,
            labels,
            groups=groups
        )
    )

    x_train = features.iloc[train_indices]
    x_test = features.iloc[test_indices]

    y_train = labels.iloc[train_indices]
    y_test = labels.iloc[test_indices]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model.fit(
        x_train,
        y_train
    )

    predictions = model.predict(
        x_test
    )

    print("Dataset shape:", dataset.shape)
    print("Training records:", len(x_train))
    print("Test records:", len(x_test))

    print(
        "Training domains:",
        groups.iloc[train_indices].nunique()
    )

    print(
        "Test domains:",
        groups.iloc[test_indices].nunique()
    )

    print("\nClass distribution:")
    print(
        labels.value_counts().sort_index()
    )

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "legitimate",
                "phishing"
            ],
            digits=4
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    print("\nMost important features:")

    importance = sorted(
        zip(
            feature_names(),
            model.feature_importances_
        ),
        key=lambda item: item[1],
        reverse=True
    )

    for name, value in importance[:10]:
        print(
            f"{name}: {value:.4f}"
        )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        {
            "model": model,
            "feature_names": list(features.columns)
        },
        MODEL_PATH
    )

    print(
        f"\nModel saved to: {MODEL_PATH}"
    )


if __name__ == "__main__":
    main()