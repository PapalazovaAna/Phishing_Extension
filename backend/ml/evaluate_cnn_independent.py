from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from ml.benchmark_deep_models import CNN, URLDataset


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "app" / "ml" / "data"
MODEL_PATH = BASE_DIR / "ml" / "cnn_url_model.pt"
DATA_PATH = DATA_DIR / "independent_evaluation_urls.csv"
OUTPUT_PATH = BASE_DIR / "ml" / "cnn_independent_predictions.csv"


def main() -> None:
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    vocabulary = checkpoint["vocabulary"]
    model = CNN(len(vocabulary) + 2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    frame = pd.read_csv(DATA_PATH)
    loader = DataLoader(
        URLDataset(frame["url"].tolist(), frame["label"], vocabulary),
        batch_size=256,
    )

    predictions = []
    probabilities = []
    with torch.no_grad():
        for features, _ in loader:
            scores = torch.sigmoid(model(features)).numpy()
            probabilities.extend(scores.tolist())
            predictions.extend((scores >= 0.5).astype(int).tolist())

    frame["phishing_probability"] = probabilities
    frame["prediction"] = predictions
    frame.to_csv(OUTPUT_PATH, index=False)

    print(classification_report(
        frame["label"],
        frame["prediction"],
        target_names=["legitimate", "phishing"],
        digits=4,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(frame["label"], frame["prediction"]))
    print(f"Predictions saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
