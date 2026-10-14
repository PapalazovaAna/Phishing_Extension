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
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_NAME = (
    "CrabInHoney/"
    "urlbert-tiny-v4-phishing-classifier"
)

DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "ml"
    / "data"
    / "vonDataset20180426.dill"
)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "urlbert_vonpower_predictions.csv"
)

MAX_PER_CLASS = 10_000
BATCH_SIZE = 64
CHUNK_OVERLAP = 16
RANDOM_SEED = 42


def labels_to_1d(labels):
    return np.asarray(labels).reshape(-1)


def decode_urls(encoded_urls, char_to_int):
    reverse_mapping = {
        int(value): str(key)
        for key, value in char_to_int.items()
    }

    decoded_urls = []

    for row in encoded_urls:
        characters = []

        for token in row:
            token = int(token)

            if token == 0:
                continue

            characters.append(
                reverse_mapping.get(token, "")
            )

        decoded_urls.append("".join(characters))

    return decoded_urls


def select_balanced_subset(urls, labels):
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

    selected_indices = np.asarray(selected_indices)

    selected_urls = [
        urls[index]
        for index in selected_indices
    ]

    selected_labels = labels[selected_indices]

    return selected_urls, selected_labels


def create_url_chunks(
    url,
    tokenizer,
    model_max_length,
):
    token_ids = tokenizer.encode(
        url,
        add_special_tokens=False,
        truncation=False,
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

    input_ids = []
    attention_masks = []

    pad_token_id = tokenizer.pad_token_id

    if pad_token_id is None:
        pad_token_id = 0

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

    for chunk in chunks:
        ids = [
            cls_token_id,
            *chunk,
            sep_token_id,
        ]

        mask = [1] * len(ids)

        padding_length = (
            model_max_length - len(ids)
        )

        ids.extend(
            [pad_token_id] * padding_length
        )

        mask.extend(
            [0] * padding_length
        )

        input_ids.append(ids)
        attention_masks.append(mask)

    return (
        torch.tensor(
            input_ids,
            dtype=torch.long,
        ),
        torch.tensor(
            attention_masks,
            dtype=torch.long,
        ),
    )


def predict_url_probability(
    url,
    model,
    tokenizer,
    model_max_length,
):
    input_ids, attention_masks = create_url_chunks(
        url,
        tokenizer,
        model_max_length,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
        )

    # Mean logits across all URL chunks
    mean_logits = outputs.logits.mean(dim=0)

    probabilities = torch.softmax(
        mean_logits,
        dim=0,
    )

    return float(probabilities[1])


def main():
    print("Loading vonpower test set...")

    with DATASET_PATH.open("rb") as file:
        data = dill.load(file)

    test_x = np.asarray(data["test_x"])
    test_y = labels_to_1d(data["test_y"])
    char_to_int = data["char_to_int"]

    test_urls = decode_urls(
        test_x,
        char_to_int,
    )

    test_urls, test_labels = select_balanced_subset(
        test_urls,
        test_y,
    )

    print(
        f"Evaluation records: {len(test_urls)}"
    )
    print(
        f"Legitimate records: "
        f"{np.sum(test_labels == 0)}"
    )
    print(
        f"Phishing records: "
        f"{np.sum(test_labels == 1)}"
    )

    print("\nLoading URLBERT...")

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

    predictions = []
    probabilities = []

    for index, url in enumerate(test_urls):
        probability = predict_url_probability(
            url,
            model,
            tokenizer,
            model_max_length,
        )

        probabilities.append(probability)
        predictions.append(
            int(probability >= 0.50)
        )

        if (index + 1) % 500 == 0:
            print(
                f"Processed "
                f"{index + 1}/{len(test_urls)}"
            )

    predictions = np.asarray(predictions)
    probabilities = np.asarray(probabilities)

    print("\nURLBERT ON VONPOWER")
    print("=" * 70)

    print(
        classification_report(
            test_labels,
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
            test_labels,
            predictions,
        )
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            test_labels,
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
        f"\nPredictions saved to: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()