from pathlib import Path
import argparse
import csv
import json
import time

import dill
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_NAME = (
    "CrabInHoney/"
    "urlbert-tiny-v4-phishing-classifier"
)

BASE_DIR = Path(__file__).resolve().parents[1]

VONPOWER_PATH = (
    BASE_DIR
    / "app"
    / "ml"
    / "data"
    / "vonDataset20180426.dill"
)

EALVARADOB_PATH = (
    BASE_DIR
    / "app"
    / "ml"
    / "data"
    / "ealvaradob"
    / "urls.json"
)

TRAIN_URLS_PER_CLASS = 5_000
TARGET_URLS_PER_CLASS = 10_000

BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 2e-5
CHUNK_OVERLAP = 16
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


def load_ealvaradob_records():
    with EALVARADOB_PATH.open(
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

    urls = np.asarray(
        list(unique_records.keys())
    )

    labels = np.asarray(
        list(unique_records.values()),
        dtype=np.int64,
    )

    return urls, labels


def load_vonpower_records():
    with VONPOWER_PATH.open("rb") as file:
        data = dill.load(file)

    train_urls = decode_vonpower_urls(
        data["train_x"],
        data["char_to_int"],
    )

    train_labels = np.asarray(
        data["train_y"]
    ).reshape(-1)

    test_urls = decode_vonpower_urls(
        data["test_x"],
        data["char_to_int"],
    )

    test_labels = np.asarray(
        data["test_y"]
    ).reshape(-1)

    return (
        np.asarray(train_urls),
        train_labels,
        np.asarray(test_urls),
        test_labels,
    )


def select_balanced(
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


def create_chunks(
    url,
    tokenizer,
    model_max_length,
):
    tokens = tokenizer.tokenize(url)

    token_ids = tokenizer.convert_tokens_to_ids(
        tokens
    )

    content_length = model_max_length - 2

    if len(token_ids) <= content_length:
        chunks = [token_ids]
    else:
        step = content_length - CHUNK_OVERLAP
        chunks = []

        for start in range(
            0,
            len(token_ids),
            step,
        ):
            chunk = token_ids[
                start:start + content_length
            ]

            if chunk:
                chunks.append(chunk)

            if start + content_length >= len(token_ids):
                break

    all_input_ids = []
    all_attention_masks = []

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id or 0

    for chunk in chunks:
        input_ids = [
            cls_token_id,
            *chunk,
            sep_token_id,
        ]

        attention_mask = [1] * len(input_ids)

        padding_length = (
            model_max_length - len(input_ids)
        )

        input_ids.extend(
            [pad_token_id] * padding_length
        )

        attention_mask.extend(
            [0] * padding_length
        )

        all_input_ids.append(input_ids)
        all_attention_masks.append(
            attention_mask
        )

    return (
        all_input_ids,
        all_attention_masks,
    )


def build_training_dataset(
    urls,
    labels,
    tokenizer,
    model_max_length,
):
    all_input_ids = []
    all_attention_masks = []
    all_labels = []

    for url, label in zip(urls, labels):
        input_ids, attention_masks = create_chunks(
            url,
            tokenizer,
            model_max_length,
        )

        all_input_ids.extend(input_ids)
        all_attention_masks.extend(attention_masks)
        all_labels.extend(
            [int(label)] * len(input_ids)
        )

    return TensorDataset(
        torch.tensor(
            all_input_ids,
            dtype=torch.long,
        ),
        torch.tensor(
            all_attention_masks,
            dtype=torch.long,
        ),
        torch.tensor(
            all_labels,
            dtype=torch.long,
        ),
    )


def train_model(
    model,
    dataset,
):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    model.train()

    for epoch in range(EPOCHS):
        total_loss = 0.0

        for input_ids, attention_masks, labels in loader:
            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
                labels=labels,
            )

            loss = outputs.loss
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = (
            total_loss / len(loader)
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS}, "
            f"loss: {average_loss:.4f}"
        )


def predict_urls(
    model,
    tokenizer,
    urls,
    model_max_length,
):
    model.eval()

    predictions = []
    probabilities = []

    with torch.no_grad():
        for index, url in enumerate(urls):
            input_ids, attention_masks = (
                create_chunks(
                    url,
                    tokenizer,
                    model_max_length,
                )
            )

            input_ids = torch.tensor(
                input_ids,
                dtype=torch.long,
            )

            attention_masks = torch.tensor(
                attention_masks,
                dtype=torch.long,
            )

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_masks,
            )

            # Mean logits across all chunks
            mean_logits = outputs.logits.mean(
                dim=0
            )

            probs = torch.softmax(
                mean_logits,
                dim=0,
            )

            phishing_probability = float(
                probs[1]
            )

            probabilities.append(
                phishing_probability
            )

            predictions.append(
                int(phishing_probability >= 0.5)
            )

            if (index + 1) % 500 == 0:
                print(
                    f"Processed "
                    f"{index + 1}/{len(urls)}"
                )

    return (
        np.asarray(predictions),
        np.asarray(probabilities),
    )


def run_experiment(train_on):
    rng = np.random.default_rng(
        RANDOM_SEED
    )

    print(
        f"\nFine-tuning URLBERT on: "
        f"{train_on}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(MODEL_NAME)
    )

    model.eval()

    model_max_length = int(
        model.config.max_position_embeddings
    )

    print(
        f"Model maximum length: "
        f"{model_max_length}"
    )

    if train_on == "vonpower":
        source_train_urls, source_train_labels, _, _ = (
            load_vonpower_records()
        )

        target_urls, target_labels = (
            load_ealvaradob_records()
        )

        source_train_urls, source_train_labels = (
            select_balanced(
                source_train_urls,
                source_train_labels,
                TRAIN_URLS_PER_CLASS,
                rng,
            )
        )

        target_urls, target_labels = (
            select_balanced(
                target_urls,
                target_labels,
                TARGET_URLS_PER_CLASS,
                rng,
            )
        )

    else:
        all_urls, all_labels = (
            load_ealvaradob_records()
        )

        source_indices, _ = train_test_split(
            np.arange(len(all_urls)),
            test_size=0.20,
            stratify=all_labels,
            random_state=RANDOM_SEED,
        )

        source_train_urls, source_train_labels = (
            select_balanced(
                all_urls[source_indices],
                all_labels[source_indices],
                TRAIN_URLS_PER_CLASS,
                rng,
            )
        )

        _, _, target_urls, target_labels = (
            load_vonpower_records()
        )

        target_urls, target_labels = (
            select_balanced(
                target_urls,
                target_labels,
                TARGET_URLS_PER_CLASS,
                rng,
            )
        )

    print(
        f"Training URLs: "
        f"{len(source_train_urls)}"
    )

    print(
        f"Target test URLs: "
        f"{len(target_urls)}"
    )

    print("\nCreating training chunks...")

    training_dataset = build_training_dataset(
        source_train_urls,
        source_train_labels,
        tokenizer,
        model_max_length,
    )

    print(
        f"Training chunks: "
        f"{len(training_dataset)}"
    )

    print("\nFine-tuning URLBERT...")

    start_time = time.time()

    train_model(
        model,
        training_dataset,
    )

    training_seconds = time.time() - start_time

    print("\nEvaluating target dataset...")

    predictions, probabilities = predict_urls(
        model,
        tokenizer,
        target_urls,
        model_max_length,
    )

    print(
        f"\nURLBERT fine-tuned on "
        f"{train_on}, tested on external dataset"
    )
    print("=" * 70)

    print(
        classification_report(
            target_labels,
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
            target_labels,
            predictions,
        )
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            target_labels,
            predictions,
        )
    )

    output_dir = (
        BASE_DIR
        / "ml"
        / f"urlbert_finetuned_{train_on}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    predictions_path = (
        BASE_DIR
        / "ml"
        / f"urlbert_{train_on}_external_predictions.csv"
    )

    with predictions_path.open(
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
            target_urls,
            target_labels,
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

    print(
        f"Model saved to: {output_dir}"
    )

    print(
        f"Predictions saved to: "
        f"{predictions_path}"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-on",
        choices=[
            "vonpower",
            "ealvaradob",
        ],
        required=True,
    )

    args = parser.parse_args()

    run_experiment(args.train_on)


if __name__ == "__main__":
    main()