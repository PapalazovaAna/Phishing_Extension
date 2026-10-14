from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, recall_score


BASE_DIR = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = BASE_DIR / "ml" / "cnn_independent_predictions.csv"


def main():
    frame = pd.read_csv(PREDICTIONS_PATH)
    labels = frame["label"].astype(int)
    probabilities = frame["phishing_probability"]

    print("Probability quantiles by true label")
    print(frame.groupby("label")["phishing_probability"].quantile(
        [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).unstack().to_string())

    rows = []
    for threshold in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99, 0.995, 0.999]:
        predictions = (probabilities >= threshold).astype(int)
        matrix = confusion_matrix(labels, predictions, labels=[0, 1])
        rows.append({
            "threshold": threshold,
            "legitimate_recall": recall_score(
                labels, predictions, pos_label=0, zero_division=0
            ),
            "phishing_recall": recall_score(
                labels, predictions, pos_label=1, zero_division=0
            ),
            "phishing_f1": f1_score(labels, predictions, zero_division=0),
            "false_positives": int(matrix[0, 1]),
            "false_negatives": int(matrix[1, 0]),
        })

    print("\nThreshold comparison")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
