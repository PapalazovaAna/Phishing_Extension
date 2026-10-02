from pathlib import Path
import sys

import joblib
import pandas as pd
import tldextract

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


LEGITPHISH_PATH = (
    CURRENT_DIRECTORY
    / "data"
    / "url_features_extracted1.csv"
)

PHIUSIIL_PATH = (
    CURRENT_DIRECTORY
    / "data"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)


def get_registered_domain(url: str) -> str:
    extracted = tldextract.extract(url)

    registered_domain = (
        extracted.top_domain_under_public_suffix
    )

    if registered_domain:
        return registered_domain.lower()

    return url.lower()


def load_legitphish() -> pd.DataFrame:
    dataframe = pd.read_csv(LEGITPHISH_PATH)

    dataframe = dataframe[
        ["URL", "ClassLabel"]
    ].dropna()

    dataframe = dataframe.rename(
        columns={
            "ClassLabel": "raw_label",
        }
    )

    dataframe["raw_label"] = (
        dataframe["raw_label"].astype(int)
    )

    # LegitPhish:
    # 0 = phishing
    # 1 = legitimate
    dataframe["target"] = (
        dataframe["raw_label"] == 0
    ).astype(int)

    dataframe["source"] = "LegitPhish"

    return dataframe[
        [
            "URL",
            "target",
            "source",
        ]
    ]


def load_phiusii() -> pd.DataFrame:
    dataframe = pd.read_csv(PHIUSIIL_PATH)

    dataframe = dataframe[
        ["URL", "label"]
    ].dropna()

    dataframe = dataframe.rename(
        columns={
            "label": "raw_label",
        }
    )

    dataframe["raw_label"] = (
        dataframe["raw_label"].astype(int)
    )

    # PhiUSIIL:
    # 0 = phishing
    # 1 = legitimate
    dataframe["target"] = (
        dataframe["raw_label"] == 0
    ).astype(int)

    dataframe["source"] = "PhiUSIIL"

    return dataframe[
        [
            "URL",
            "target",
            "source",
        ]
    ]


def prepare_dataset() -> pd.DataFrame:
    legitphish = load_legitphish()
    phiusii = load_phiusii()

    dataframe = pd.concat(
        [
            legitphish,
            phiusii,
        ],
        ignore_index=True,
    )

    dataframe["URL"] = dataframe["URL"].astype(str)

    # Remove exact duplicate URLs before splitting.
    # This prevents the same URL from appearing
    # in both training and testing data.
    dataframe = dataframe.drop_duplicates(
        subset=["URL"]
    )

    dataframe["registered_domain"] = (
        dataframe["URL"].apply(
            get_registered_domain
        )
    )

    return dataframe


def create_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            extract_legitphish_features(url)
            for url in dataframe["URL"]
        ]
    )


def create_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
            min_samples_leaf=2,
            max_features="sqrt",
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300,
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


def print_source_results(
    model,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    test_metadata: pd.DataFrame,
    source: str,
) -> None:
    source_indices = (
        test_metadata["source"].to_numpy()
        == source
    )

    if source_indices.sum() == 0:
        return

    source_features = x_test.iloc[
        source_indices
    ]

    source_target = y_test.iloc[
        source_indices
    ]

    source_predictions = model.predict(
        source_features
    )

    print(f"\nResults for {source}:")
    print(
        classification_report(
            source_target,
            source_predictions,
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
            source_target,
            source_predictions,
        )
    )


def main() -> None:
    dataframe = prepare_dataset()

    feature_dataframe = create_features(
        dataframe
    )

    feature_columns = feature_names()

    feature_dataframe = feature_dataframe[
        feature_columns
    ]

    target = dataframe["target"]

    groups = dataframe["registered_domain"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=42,
    )

    train_indices, test_indices = next(
        splitter.split(
            feature_dataframe,
            target,
            groups=groups,
        )
    )

    x_train = feature_dataframe.iloc[
        train_indices
    ]

    x_test = feature_dataframe.iloc[
        test_indices
    ]

    y_train = target.iloc[train_indices]
    y_test = target.iloc[test_indices]

    train_metadata = dataframe.iloc[
        train_indices
    ]

    test_metadata = dataframe.iloc[
        test_indices
    ]

    print("Combined dataset shape:")
    print(dataframe.shape)

    print("\nSource distribution:")
    print(
        dataframe["source"].value_counts()
    )

    print("\nClass distribution:")
    print(
        target.map(
            {
                0: "legitimate",
                1: "phishing",
            }
        ).value_counts()
    )

    print(
        f"\nTraining records: {len(x_train)}"
    )

    print(
        f"Test records: {len(x_test)}"
    )

    print(
        f"Training domains: "
        f"{train_metadata['registered_domain'].nunique()}"
    )

    print(
        f"Test domains: "
        f"{test_metadata['registered_domain'].nunique()}"
    )

    models = create_models()
    model_results = []

    for model_name, model in models.items():
        print("\n" + "=" * 70)
        print(model_name)
        print("=" * 70)

        model.fit(x_train, y_train)

        predictions = model.predict(x_test)

        print("\nOverall classification report:")
        print(
            classification_report(
                y_test,
                predictions,
                target_names=[
                    "legitimate",
                    "phishing",
                ],
                digits=4,
            )
        )

        print("Overall accuracy:")
        print(
            f"{accuracy_score(y_test, predictions):.4f}"
        )

        print("\nOverall confusion matrix:")
        print(
            confusion_matrix(
                y_test,
                predictions,
            )
        )

        print_source_results(
            model=model,
            x_test=x_test,
            y_test=y_test,
            test_metadata=test_metadata,
            source="LegitPhish",
        )

        print_source_results(
            model=model,
            x_test=x_test,
            y_test=y_test,
            test_metadata=test_metadata,
            source="PhiUSIIL",
        )

        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
        )

        phishing_f1 = report["1"]["f1-score"]
        legitimate_recall = report["0"]["recall"]

        # We prefer a model that performs well on phishing
        # while still maintaining good legitimate recall.
        combined_score = (
            phishing_f1 * 0.70
            + legitimate_recall * 0.30
        )

        model_results.append(
            {
                "name": model_name,
                "model": model,
                "phishing_f1": phishing_f1,
                "legitimate_recall": legitimate_recall,
                "combined_score": combined_score,
            }
        )

    model_results.sort(
        key=lambda result: result["combined_score"],
        reverse=True,
    )

    best_result = model_results[0]

    artifact = {
        "model": best_result["model"],
        "feature_names": feature_columns,
        "model_name": best_result["name"],
        "phishing_f1": best_result["phishing_f1"],
        "legitimate_recall": (
            best_result["legitimate_recall"]
        ),
        "combined_score": (
            best_result["combined_score"]
        ),
    }

    output_path = (
        CURRENT_DIRECTORY
        / "combined_url_model.joblib"
    )

    joblib.dump(artifact, output_path)

    print("\n" + "=" * 70)
    print("BEST COMBINED MODEL")
    print("=" * 70)
    print(
        f"Model: {best_result['name']}"
    )
    print(
        f"Phishing F1: "
        f"{best_result['phishing_f1']:.4f}"
    )
    print(
        f"Legitimate recall: "
        f"{best_result['legitimate_recall']:.4f}"
    )
    print(
        f"Combined score: "
        f"{best_result['combined_score']:.4f}"
    )
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()