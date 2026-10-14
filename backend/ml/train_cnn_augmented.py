from pathlib import Path
from urllib.parse import urlparse
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from ml.benchmark_deep_models import (
    CNN,
    URLDataset,
    load_dataset,
    metrics,
    train_model,
)


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "app" / "ml" / "data"

ORIGINAL_HOLDOUT_PATH = DATA_DIR / "external_evaluation_urls.csv"
INDEPENDENT_PATH = DATA_DIR / "independent_evaluation_urls.csv"

MODEL_PATH = BASE_DIR / "ml" / "cnn_augmented_model.pt"
PREDICTIONS_PATH = BASE_DIR / "ml" / "cnn_augmented_predictions.csv"

SEED = 42
MAX_TRAIN_PER_CLASS = 50_000
BATCH_SIZE = 256


def registered_domain(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower().strip(".")
    parts = hostname.split(".")

    return ".".join(parts[-2:]) if len(parts) >= 2 else hostname


def load_original_data() -> pd.DataFrame:
    dataset_paths = [
        DATA_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv",
        DATA_DIR / "url_features_extracted1.csv",
    ]

    datasets = [
        load_dataset(path)
        for path in dataset_paths
        if path.exists()
    ]

    frame = pd.concat(datasets, ignore_index=True)
    frame = frame.drop_duplicates("url").reset_index(drop=True)

    if ORIGINAL_HOLDOUT_PATH.exists():
        original_holdout = pd.read_csv(ORIGINAL_HOLDOUT_PATH)
        excluded_urls = set(original_holdout["url"].astype(str))

        frame = frame[
            ~frame["url"].astype(str).isin(excluded_urls)
        ].copy()

    frame["domain"] = frame["url"].map(registered_domain)

    return frame.reset_index(drop=True)


def split_by_domain(frame: pd.DataFrame):
    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=SEED,
    )

    train_indices, test_indices = next(
        splitter.split(
            frame["url"],
            frame["label"],
            groups=frame["domain"],
        )
    )

    return (
        frame.iloc[train_indices].copy(),
        frame.iloc[test_indices].copy(),
    )


def balanced_sample(frame: pd.DataFrame) -> pd.DataFrame:
    parts = []

    for _, group in frame.groupby("label"):
        sample_size = min(
            len(group),
            MAX_TRAIN_PER_CLASS,
        )

        parts.append(
            group.sample(
                sample_size,
                random_state=SEED,
            )
        )

    return pd.concat(parts, ignore_index=True)


def build_vocabulary(urls) -> dict[str, int]:
    characters = sorted(
        set("".join(urls))
    )

    return {
        character: index + 2
        for index, character in enumerate(characters)
    }


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    print("Loading original datasets...")
    original = load_original_data()

    original_train, original_test = split_by_domain(original)
    original_train = balanced_sample(original_train)

    print(f"Original training records: {len(original_train)}")
    print(f"Original test records: {len(original_test)}")

    independent = pd.read_csv(INDEPENDENT_PATH)
    independent["url"] = independent["url"].astype(str)
    independent["label"] = independent["label"].astype(int)
    independent["domain"] = independent["url"].map(
        registered_domain
    )

    independent_augmented, independent_final = split_by_domain(
        independent
    )

    print(
        "Independent augmentation records:",
        len(independent_augmented),
    )
    print(
        "Independent final test records:",
        len(independent_final),
    )

    augmented_train = pd.concat(
        [
            original_train,
            independent_augmented,
        ],
        ignore_index=True,
    )

    augmented_train = augmented_train.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    vocabulary = build_vocabulary(
        augmented_train["url"].tolist()
    )

    train_dataset = URLDataset(
        augmented_train["url"].tolist(),
        augmented_train["label"].tolist(),
        vocabulary,
    )

    original_test_dataset = URLDataset(
        original_test["url"].tolist(),
        original_test["label"].tolist(),
        vocabulary,
    )

    independent_final_dataset = URLDataset(
        independent_final["url"].tolist(),
        independent_final["label"].tolist(),
        vocabulary,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    original_test_loader = DataLoader(
        original_test_dataset,
        batch_size=BATCH_SIZE,
    )

    independent_final_loader = DataLoader(
        independent_final_dataset,
        batch_size=BATCH_SIZE,
    )

    model = CNN(len(vocabulary) + 2)

    print("\nTraining augmented CNN...")
    started = perf_counter()

    train_model(
        model,
        train_loader,
    )

    training_seconds = perf_counter() - started

    print("\nOriginal domain-aware test:")
    original_metrics = metrics(
        model,
        original_test_loader,
    )
    print(original_metrics)

    print("\nIndependent final test:")
    independent_metrics = metrics(
        model,
        independent_final_loader,
    )
    print(independent_metrics)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocabulary": vocabulary,
        },
        MODEL_PATH,
    )

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Training time: {training_seconds:.2f} seconds")

    model.eval()
    predictions = []

    with torch.no_grad():
        for features, _ in independent_final_loader:
            probabilities = torch.sigmoid(
                model(features)
            ).numpy()

            predictions.extend(
                (probabilities >= 0.5).astype(int)
            )

    result = independent_final[
        ["url", "label"]
    ].copy()

    result["prediction"] = predictions

    result.to_csv(
        PREDICTIONS_PATH,
        index=False,
    )

    print(
        f"Predictions saved to: {PREDICTIONS_PATH}"
    )


if __name__ == "__main__":
    main()