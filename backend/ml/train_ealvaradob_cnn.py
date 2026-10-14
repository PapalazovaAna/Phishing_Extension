from pathlib import Path
import csv
import json
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "ml"
    / "data"
    / "ealvaradob"
    / "urls.json"
)

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "ealvaradob_cnn_model.pt"
)

PREDICTIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "ealvaradob_cnn_predictions.csv"
)

MAX_URL_LENGTH = 150
MAX_TRAIN_PER_CLASS = 100_000
MAX_VALIDATION_PER_CLASS = 5_000
MAX_TEST_PER_CLASS = 10_000

BATCH_SIZE = 512
EPOCHS = 2
LEARNING_RATE = 0.001
RANDOM_SEED = 42


class URLCNN(nn.Module):
    def __init__(self, vocabulary_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size + 1,
            embedding_dim=32,
            padding_idx=0,
        )

        self.convolution = nn.Conv1d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            padding=2,
        )

        self.activation = nn.ReLU()
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(64, 1)

    def forward(self, inputs):
        embedded = self.embedding(inputs)
        embedded = embedded.transpose(1, 2)

        features = self.convolution(embedded)
        features = self.activation(features)
        features = self.pooling(features)
        features = features.squeeze(-1)

        return self.classifier(features).squeeze(1)


def load_unique_records():
    print("Loading ealvaradob dataset...")

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        records = json.load(file)

    unique_records = {}

    for record in records:
        url = str(record["text"]).strip()
        label = int(record["label"])

        if url:
            unique_records[url] = label

    urls = np.asarray(list(unique_records.keys()))
    labels = np.asarray(
        list(unique_records.values()),
        dtype=np.int64,
    )

    print(f"Unique records: {len(urls)}")
    print(
        f"Legitimate: "
        f"{np.sum(labels == 0)}"
    )
    print(
        f"Phishing: "
        f"{np.sum(labels == 1)}"
    )

    return urls, labels


def create_character_mapping(urls):
    characters = sorted(
        set(
            character
            for url in urls
            for character in url
        )
    )

    # 0 is reserved for padding
    char_to_int = {
        character: index + 1
        for index, character in enumerate(characters)
    }

    return char_to_int


def encode_urls(urls, char_to_int):
    encoded_urls = np.zeros(
        (
            len(urls),
            MAX_URL_LENGTH,
        ),
        dtype=np.int64,
    )

    for row_index, url in enumerate(urls):
        encoded = [
            char_to_int.get(character, 0)
            for character in url[:MAX_URL_LENGTH]
        ]

        encoded_urls[
            row_index,
            :len(encoded),
        ] = encoded

    return encoded_urls


def select_per_class(
    urls,
    labels,
    max_per_class,
    rng,
):
    selected_indices = []

    for label in [0, 1]:
        label_indices = np.flatnonzero(
            labels == label
        )

        sample_size = min(
            max_per_class,
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

    return (
        urls[selected_indices],
        labels[selected_indices],
    )


def create_loader(
    features,
    labels,
    shuffle,
):
    x_tensor = torch.tensor(
        features,
        dtype=torch.long,
    )

    y_tensor = torch.tensor(
        labels,
        dtype=torch.float32,
    )

    dataset = TensorDataset(
        x_tensor,
        y_tensor,
    )

    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=0,
    )


def evaluate(model, features, labels):
    loader = create_loader(
        features,
        labels,
        shuffle=False,
    )

    predictions = []
    probabilities = []

    model.eval()

    with torch.no_grad():
        for inputs, _ in loader:
            logits = model(inputs)
            probs = torch.sigmoid(logits)

            probabilities.extend(
                probs.cpu().numpy().tolist()
            )

            predictions.extend(
                (probs >= 0.5)
                .int()
                .cpu()
                .numpy()
                .tolist()
            )

    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

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

    return predictions, probabilities


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    urls, labels = load_unique_records()

    char_to_int = create_character_mapping(
        urls
    )

    print(
        f"Character vocabulary size: "
        f"{len(char_to_int)}"
    )

    all_indices = np.arange(len(urls))

    train_indices, temporary_indices = (
        train_test_split(
            all_indices,
            test_size=0.20,
            stratify=labels,
            random_state=RANDOM_SEED,
        )
    )

    validation_indices, test_indices = (
        train_test_split(
            temporary_indices,
            test_size=0.50,
            stratify=labels[temporary_indices],
            random_state=RANDOM_SEED,
        )
    )

    train_urls, train_labels = select_per_class(
        urls[train_indices],
        labels[train_indices],
        MAX_TRAIN_PER_CLASS,
        rng,
    )

    validation_urls, validation_labels = (
        select_per_class(
            urls[validation_indices],
            labels[validation_indices],
            MAX_VALIDATION_PER_CLASS,
            rng,
        )
    )

    test_urls, test_labels = select_per_class(
        urls[test_indices],
        labels[test_indices],
        MAX_TEST_PER_CLASS,
        rng,
    )

    print(
        f"\nTraining records: "
        f"{len(train_urls)}"
    )
    print(
        f"Validation records: "
        f"{len(validation_urls)}"
    )
    print(
        f"Test records: "
        f"{len(test_urls)}"
    )

    print("\nEncoding URLs...")

    train_x = encode_urls(
        train_urls,
        char_to_int,
    )

    validation_x = encode_urls(
        validation_urls,
        char_to_int,
    )

    test_x = encode_urls(
        test_urls,
        char_to_int,
    )

    print("\nTraining ealvaradob CNN...")

    train_loader = create_loader(
        train_x,
        train_labels,
        shuffle=True,
    )

    model = URLCNN(
        vocabulary_size=len(char_to_int)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    loss_function = nn.BCEWithLogitsLoss()

    start_time = time.time()

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for inputs, batch_labels in train_loader:
            optimizer.zero_grad()

            logits = model(inputs)
            loss = loss_function(
                logits,
                batch_labels,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(train_loader)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"loss: {average_loss:.4f}"
        )

    training_seconds = time.time() - start_time

    print("\nValidation results")
    print("=" * 70)

    evaluate(
        model,
        validation_x,
        validation_labels,
    )

    print("\nTest results")
    print("=" * 70)

    predictions, probabilities = evaluate(
        model,
        test_x,
        test_labels,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "char_to_int": char_to_int,
            "sequence_length": MAX_URL_LENGTH,
            "vocabulary_size": len(char_to_int),
        },
        MODEL_PATH,
    )

    with PREDICTIONS_PATH.open(
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
            test_urls,
            test_labels,
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
        f"\nTraining time: "
        f"{training_seconds:.2f} seconds"
    )
    print(f"Model saved to: {MODEL_PATH}")
    print(
        f"Predictions saved to: "
        f"{PREDICTIONS_PATH}"
    )


if __name__ == "__main__":
    main()