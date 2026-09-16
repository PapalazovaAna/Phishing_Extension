from pathlib import Path

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

from train import (
    find_column,
    convert_labels,
    domain_from_url
)


DATASET_PATH = Path(
    "app/ml/data/PhiUSIIL_Phishing_URL_Dataset.csv"
)


def evaluate_model(
    features: pd.DataFrame,
    labels: pd.Series,
    groups: pd.Series,
    model_name: str
) -> None:
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

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

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

    print("\nTop features:")

    importance = sorted(
        zip(
            features.columns,
            model.feature_importances_
        ),
        key=lambda item: item[1],
        reverse=True
    )

    for name, value in importance[:10]:
        print(
            f"{name}: {value:.4f}"
        )


def main() -> None:
    dataset = pd.read_csv(
        DATASET_PATH
    )

    columns = list(
        dataset.columns
    )

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

    features = pd.DataFrame(
        dataset["url"]
        .map(extract_url_features)
        .tolist(),
        columns=feature_names()
    )

    labels = convert_labels(
        dataset["label"]
    )

    groups = dataset["url"].map(
        domain_from_url
    )

    print("Dataset shape:")
    print(dataset.shape)

    print("\nHTTPS distribution by class:")

    https_distribution = pd.crosstab(
        labels,
        features["has_https"],
        normalize="index"
    )

    https_distribution.index = [
        "legitimate",
        "phishing"
    ]

    https_distribution.columns = [
        "http",
        "https"
    ]

    print(
        https_distribution
    )

    evaluate_model(
        features,
        labels,
        groups,
        "Model with all URL features"
    )

    features_without_https = features.drop(
        columns=["has_https"]
    )

    evaluate_model(
        features_without_https,
        labels,
        groups,
        "Model without has_https"
    )


if __name__ == "__main__":
    main()