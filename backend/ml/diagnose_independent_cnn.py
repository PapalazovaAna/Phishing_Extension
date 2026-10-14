from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from ml.benchmark_deep_models import CNN, URLDataset


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "app" / "ml" / "data"
MODEL_PATH = BASE_DIR / "ml" / "cnn_url_model.pt"
DATA_PATH = DATA_DIR / "independent_evaluation_urls.csv"


def main():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    vocabulary = checkpoint["vocabulary"]
    model = CNN(len(vocabulary) + 2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    frame = pd.read_csv(DATA_PATH)
    frame["url"] = frame["url"].astype(str)
    frame["url_length"] = frame["url"].str.len()
    frame["unknown_character_ratio"] = frame["url"].map(
        lambda url: sum(character not in vocabulary for character in url)
        / max(len(url), 1)
    )

    loader = DataLoader(
        URLDataset(frame["url"].tolist(), frame["label"], vocabulary),
        batch_size=256,
    )

    probabilities = []
    with torch.no_grad():
        for features, _ in loader:
            probabilities.extend(torch.sigmoid(model(features)).tolist())

    frame["phishing_probability"] = probabilities
    frame["prediction"] = (frame["phishing_probability"] >= 0.5).astype(int)

    print("\nStatistics by true label")
    print(frame.groupby("label")[[
        "url_length",
        "unknown_character_ratio",
        "phishing_probability",
        "prediction",
    ]].agg(["count", "mean", "min", "max"]))

    print("\nLegitimate false positives with lowest scores")
    print(frame[(frame["label"] == 0) & (frame["prediction"] == 1)]
          .sort_values("phishing_probability")[[
              "url", "url_length", "unknown_character_ratio",
              "phishing_probability",
          ]].head(20).to_string(index=False))

    print("\nCharacter vocabulary size:", len(vocabulary))
    print("Saved predictions: cnn_independent_predictions.csv")


if __name__ == "__main__":
    main()
