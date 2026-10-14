from pathlib import Path
import csv

import dill
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from ml.train_ealvaradob_cnn import URLCNN


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "ealvaradob_cnn_model.pt"
)

VONPOWER_DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "ml"
    / "data"
    / "vonDataset20180426.dill"
)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "ealvaradob_cnn_vonpower_predictions.csv"
)

MAX_PER_CLASS = 10_000
RANDOM_SEED = 42


def decode_vonpower_urls(encoded_urls, char_to_int):
    reverse_mapping = {
        int(value): str(key)
        for key, value in char_to_int.items()
    }

    urls = []

    for row in encoded_urls:
        characters = []

        for token in row:
            token = int(token)

            if token == 0:
                continue

            characters.append(
                reverse_mapping.get(token, "")
            )

        urls.append("".join(characters))

    return urls


def select_balanced_subset(urls, labels):
    rng = np.random.default_rng(
        RANDOM_SEED
    )

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

    selected_indices = np.asarray(
        selected_indices
    )

    selected_urls = [
        urls[index]
        for index in selected_indices
    ]

    selected_labels = labels[
        selected_indices
    ]

    return selected_urls, selected_labels


def encode_url(
    url,
    char_to_int,
    sequence_length,
):
    encoded = [
        char_to_int.get(character, 0)
        for character in url[:sequence_length]
    ]

    if len(encoded) < sequence_length:
        encoded.extend(
            [0] * (
                sequence_length - len(encoded)
            )
        )

    return encoded


def main():
    print("Loading ealvaradob CNN...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    model = URLCNN(
        vocabulary_size=checkpoint[
            "vocabulary_size"
        ]
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    ealvaradob_char_to_int = (
        checkpoint["char_to_int"]
    )

    sequence_length = checkpoint[
        "sequence_length"
    ]

    print("Loading vonpower test set...")

    with VONPOWER_DATASET_PATH.open(
        "rb"
    ) as file:
        data = dill.load(file)

    vonpower_test_x = np.asarray(
        data["test_x"]
    )

    vonpower_test_y = np.asarray(
        data["test_y"]
    ).reshape(-1)

    vonpower_char_to_int = data[
        "char_to_int"
    ]

    decoded_urls = decode_vonpower_urls(
        vonpower_test_x,
        vonpower_char_to_int,
    )

    urls, labels = select_balanced_subset(
        decoded_urls,
        vonpower_test_y,
    )

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

    encoded_urls = np.asarray(
        [
            encode_url(
                url,
                ealvaradob_char_to_int,
                sequence_length,
            )
            for url in urls
        ],
        dtype=np.int64,
    )

    inputs = torch.tensor(
        encoded_urls,
        dtype=torch.long,
    )

    predictions = []
    probabilities = []

    batch_size = 512

    with torch.no_grad():
        for start in range(
            0,
            len(inputs),
            batch_size,
        ):
            batch = inputs[
                start:start + batch_size
            ]

            logits = model(batch)

            batch_probabilities = (
                torch.sigmoid(logits)
                .cpu()
                .numpy()
            )

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

    print("\nEALVARADOB CNN ON VONPOWER")
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