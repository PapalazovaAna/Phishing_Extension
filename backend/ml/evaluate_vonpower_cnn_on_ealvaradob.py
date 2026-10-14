from pathlib import Path
import csv
import json

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from ml.train_vonpower_cnn import URLCNN


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "vonpower_cnn_model.pt"
)

DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "ml"
    / "data"
    / "ealvaradob"
    / "urls.json"
)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "vonpower_cnn_ealvaradob_predictions.csv"
)

MAX_PER_CLASS = 10_000
RANDOM_SEED = 42


def load_balanced_records():
    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    # Remove duplicate text values
    unique_records = {}
    for record in records:
        url = str(record["text"]).strip()
        label = int(record["label"])

        if url and url not in unique_records:
            unique_records[url] = label

    urls = np.asarray(list(unique_records.keys()))
    labels = np.asarray(
        list(unique_records.values()),
        dtype=np.int64,
    )

    rng = np.random.default_rng(RANDOM_SEED)
    selected_indices = []

    for label in [0, 1]:
        label_indices = np.flatnonzero(
            labels == label
        )

        sample_size = min(
            MAX_PER_CLASS,
            len(label_indices),
        )

        selected = rng.choice(
            label_indices,
            size=sample_size,
            replace=False,
        )

        selected_indices.extend(
            selected.tolist()
        )

    rng.shuffle(selected_indices)

    return (
        urls[selected_indices],
        labels[selected_indices],
    )


def encode_url(
    url,
    char_to_int,
    sequence_length,
):
    unknown_token = char_to_int.get("�", 0)

    encoded = [
        char_to_int.get(
            character,
            unknown_token,
        )
        for character in url
    ]

    encoded = encoded[:sequence_length]

    if len(encoded) < sequence_length:
        encoded.extend(
            [0] * (
                sequence_length - len(encoded)
            )
        )

    return encoded


def main():
    print("Loading vonpower CNN...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    char_to_int = checkpoint["char_to_int"]
    sequence_length = checkpoint["sequence_length"]
    vocabulary_size = checkpoint["vocabulary_size"]

    model = URLCNN(
        vocabulary_size=vocabulary_size
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    urls, labels = load_balanced_records()

    print(
        f"Evaluation records: {len(urls)}"
    )
    print(
        f"Legitimate records: "
        f"{np.sum(labels == 0)}"
    )
    print(
        f"Phishing records: "
        f"{np.sum(labels == 1)}"
    )

    inputs = np.asarray(
        [
            encode_url(
                url,
                char_to_int,
                sequence_length,
            )
            for url in urls
        ],
        dtype=np.int64,
    )

    input_tensor = torch.tensor(
        inputs,
        dtype=torch.long,
    )

    predictions = []
    probabilities = []

    batch_size = 512

    with torch.no_grad():
        for start in range(
            0,
            len(input_tensor),
            batch_size,
        ):
            batch = input_tensor[
                start:start + batch_size
            ]

            logits = model(batch)
            batch_probabilities = torch.sigmoid(
                logits
            ).cpu().numpy()

            probabilities.extend(
                batch_probabilities.tolist()
            )

            predictions.extend(
                (batch_probabilities >= 0.5)
                .astype(int)
                .tolist()
            )

    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    print("\nVONPOWER CNN ON EALVARADOB")
    print("=" * 70)

    print(
        classification_report(
            labels,
            predictions,
            target_names=[
                "legitimate",
                "phishing",
            ],
            digits=4,
        )
    )

    print("Accuracy:")
    print(
        accuracy_score(
            labels,
            predictions,
        )
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            labels,
            predictions,
        )
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "url",
                "true_label",
                "prediction",
                "phishing_probability",
            ]
        )

        for url, label, prediction, probability in zip(
            urls,
            labels,
            predictions,
            probabilities,
        ):
            writer.writerow(
                [
                    url,
                    int(label),
                    int(prediction),
                    float(probability),
                ]
            )

    print(
        f"\nPredictions saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()