from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


BASE_DIR = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    BASE_DIR / "ml" / "cnn_augmented_predictions.csv"
)

MODEL_NAME = (
    "CrabInHoney/"
    "urlbert-tiny-v4-phishing-classifier"
)

MODEL_MAX_LENGTH = 64
CHUNK_OVERLAP = 16
BATCH_SIZE = 64


def create_url_chunks(url, tokenizer):
    token_ids = tokenizer(
        url,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    special_tokens_count = (
        tokenizer.num_special_tokens_to_add(
            pair=False
        )
    )

    content_length = (
        MODEL_MAX_LENGTH - special_tokens_count
    )

    step = content_length - CHUNK_OVERLAP
    chunks = []

    start = 0

    while start < len(token_ids):
        chunk_ids = token_ids[
            start:start + content_length
        ]

        input_ids = [
            tokenizer.cls_token_id,
            *chunk_ids,
            tokenizer.sep_token_id,
        ]

        input_ids = input_ids[:MODEL_MAX_LENGTH]

        padding_length = (
            MODEL_MAX_LENGTH - len(input_ids)
        )

        pad_token_id = tokenizer.pad_token_id

        if pad_token_id is None:
            pad_token_id = 0

        attention_mask = (
            [1] * len(input_ids)
            + [0] * padding_length
        )

        input_ids = (
            input_ids
            + [pad_token_id] * padding_length
        )

        encoded = {
            "input_ids": torch.tensor(
                [input_ids],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [attention_mask],
                dtype=torch.long,
            ),
        }

        chunks.append(encoded)
        start += step

    if not chunks:
        encoded = tokenizer(
            url,
            max_length=MODEL_MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        chunks.append(encoded)

    return chunks


def predict_url_probability(
    url,
    model,
    tokenizer,
):
    chunks = create_url_chunks(
        url,
        tokenizer,
    )

    chunk_logits = []

    with torch.no_grad():
        for chunk in chunks:
            outputs = model(**chunk)
            chunk_logits.append(
                outputs.logits.squeeze(0)
            )

    # Просекување на logits од сите URL chunks
    average_logits = torch.stack(
        chunk_logits
    ).mean(dim=0)

    probabilities = torch.softmax(
        average_logits,
        dim=0,
    )

    # Кај овој модел LABEL_1 е phishing
    return float(probabilities[1])


def main():
    print("Loading URLBERT...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(MODEL_NAME)
    )

    model.eval()

    frame = pd.read_csv(
        PREDICTIONS_PATH
    )

    urls = frame["url"].astype(str).tolist()
    labels = frame["label"].astype(int).tolist()

    probabilities = []

    for index, url in enumerate(urls):
        probability = predict_url_probability(
            url,
            model,
            tokenizer,
        )

        probabilities.append(probability)

        if (index + 1) % 100 == 0:
            print(
                f"Processed "
                f"{index + 1}/{len(urls)}"
            )

    predictions = [
        int(probability >= 0.5)
        for probability in probabilities
    ]

    print("\nURLBERT CHUNKED EVALUATION")
    print("=" * 70)
    print(
        f"Model maximum length: "
        f"{MODEL_MAX_LENGTH}"
    )
    print(
        f"Chunk overlap: "
        f"{CHUNK_OVERLAP}"
    )
    print(
        "Aggregation: mean logits across all chunks"
    )

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

    frame["urlbert_phishing_probability"] = (
        probabilities
    )

    frame["urlbert_prediction"] = predictions

    output_path = (
        BASE_DIR
        / "ml"
        / "urlbert_same_final_predictions.csv"
    )

    frame.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Predictions saved to: {output_path}"
    )


if __name__ == "__main__":
    main()