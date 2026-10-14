from pathlib import Path
import csv
import time

import dill
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "ml"
    / "data"
    / "vonDataset20180426.dill"
)

MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "vonpower_cnn_model.pt"
)

PREDICTIONS_PATH = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "vonpower_cnn_predictions.csv"
)

MAX_TRAIN_PER_CLASS = 100_000
BATCH_SIZE = 512
EPOCHS = 2
LEARNING_RATE = 0.001
RANDOM_SEED = 42


class URLCNN(nn.Module):
    def __init__(self, vocabulary_size: int):
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
        features = self.pooling(features).squeeze(-1)

        return self.classifier(features).squeeze(1)


def select_balanced_subset(features, labels):
    rng = np.random.default_rng(RANDOM_SEED)

    selected_indices = []

    for label in [0, 1]:
        label_indices = np.flatnonzero(labels == label)

        sample_size = min(
            MAX_TRAIN_PER_CLASS,
            len(label_indices),
        )

        selected = rng.choice(
            label_indices,
            size=sample_size,
            replace=False,
        )

        selected_indices.extend(selected.tolist())

    rng.shuffle(selected_indices)

    selected_indices = np.asarray(selected_indices)

    return (
        features[selected_indices],
        labels[selected_indices],
    )


def create_loader(features, labels, shuffle):
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
    print(accuracy_score(labels, predictions))

    print("\nConfusion matrix:")
    print(confusion_matrix(labels, predictions))

    return predictions, probabilities


def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("Loading vonpower dataset...")

    with DATASET_PATH.open("rb") as file:
        data = dill.load(file)

    train_x = np.asarray(data["train_x"])
    train_y = np.asarray(data["train_y"]).reshape(-1)

    val_x = np.asarray(data["val_x"])
    val_y = np.asarray(data["val_y"]).reshape(-1)

    test_x = np.asarray(data["test_x"])
    test_y = np.asarray(data["test_y"]).reshape(-1)

    char_to_int = data["char_to_int"]

    print(f"Original train records: {len(train_x)}")
    print(f"Validation records: {len(val_x)}")
    print(f"Test records: {len(test_x)}")
    print(f"Vocabulary size: {len(char_to_int)}")

    train_x, train_y = select_balanced_subset(
        train_x,
        train_y,
    )

    print(
        f"Selected training records: {len(train_x)}"
    )

    print("\nTraining CNN...")

    train_loader = create_loader(
        train_x,
        train_y,
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

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            logits = model(inputs)
            loss = loss_function(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"loss: {average_loss:.4f}"
        )

    training_seconds = time.time() - start_time

    print("\nValidation results")
    print("=" * 70)
    evaluate(model, val_x, val_y)

    print("\nTest results")
    print("=" * 70)

    predictions, probabilities = evaluate(
        model,
        test_x,
        test_y,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "char_to_int": char_to_int,
            "sequence_length": int(train_x.shape[1]),
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
                "true_label",
                "prediction",
                "phishing_probability",
            ]
        )

        for true_label, prediction, probability in zip(
            test_y,
            predictions,
            probabilities,
        ):
            writer.writerow(
                [
                    int(true_label),
                    int(prediction),
                    float(probability),
                ]
            )

    print(f"\nTraining time: {training_seconds:.2f} seconds")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Predictions saved to: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()