from pathlib import Path
import sys

import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)


CURRENT_DIRECTORY = Path(__file__).resolve().parent

if str(CURRENT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIRECTORY))

from legitphish_features import (
    extract_legitphish_features,
)


MODEL_PATH = (
    CURRENT_DIRECTORY
    / "legitphish_v2_model.joblib"
)

DATASET_PATH = (
    CURRENT_DIRECTORY
    / "data"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)


def main() -> None:
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    feature_names = artifact["feature_names"]
    model_name = artifact["model_name"]

    dataframe = pd.read_csv(DATASET_PATH)

    dataframe = dataframe[
        ["URL", "label"]
    ].dropna()

    dataframe = dataframe.drop_duplicates(
        subset=["URL"]
    )

    features = pd.DataFrame(
        [
            extract_legitphish_features(url)
            for url in dataframe["URL"]
        ]
    )

    features = features[feature_names]

    # PhiUSIIL:
    # raw label 0 = phishing
    # raw label 1 = legitimate
    target = (
        dataframe["label"] == 0
    ).astype(int)

    probabilities = model.predict_proba(
        features
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    print("=" * 70)
    print("CROSS-DATASET EVALUATION")
    print("=" * 70)
    print(f"Model: {model_name}")
    print(f"Dataset records: {len(dataframe)}")

    print("\nClassification report:")
    print(
        classification_report(
            target,
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
            target,
            predictions,
        )
    )

    print("\nThreshold comparison:")
    print("-" * 70)

    for threshold in [
        0.50,
        0.60,
        0.70,
        0.80,
        0.85,
        0.90,
    ]:
        threshold_predictions = (
            probabilities >= threshold
        ).astype(int)

        report = classification_report(
            target,
            threshold_predictions,
            output_dict=True,
        )

        phishing_metrics = report["1"]

        legitimate_metrics = report["0"]

        print(
            f"Threshold {threshold:.2f} | "
            f"Phishing precision: "
            f"{phishing_metrics['precision']:.4f} | "
            f"Phishing recall: "
            f"{phishing_metrics['recall']:.4f} | "
            f"Phishing F1: "
            f"{phishing_metrics['f1-score']:.4f} | "
            f"Legitimate recall: "
            f"{legitimate_metrics['recall']:.4f}"
        )


if __name__ == "__main__":
    main()