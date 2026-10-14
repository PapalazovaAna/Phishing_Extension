from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_DIRECTORY = (
    Path(__file__).parent
    / "models"
    / "urlbert_vonpower_full"
)

tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIRECTORY),
    local_files_only=True,
    use_fast=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIRECTORY),
    local_files_only=True,
)

model.eval()
model.to("cpu")


CHUNK_OVERLAP = 16


def classify_probability(
    probability: float,
) -> str:
    if probability >= 0.70:
        return "Likely phishing"

    if probability >= 0.40:
        return "Suspicious"

    return "Likely safe"


def create_url_chunks(url: str):
    max_length = int(
        model.config.max_position_embeddings
    )

    tokens = tokenizer.tokenize(url)

    token_ids = tokenizer.convert_tokens_to_ids(
        tokens
    )

    content_length = max_length - 2

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

            if start + content_length >= len(
                token_ids
            ):
                break

    input_ids = []
    attention_masks = []

    for chunk in chunks:
        ids = [
            tokenizer.cls_token_id,
            *chunk,
            tokenizer.sep_token_id,
        ]

        mask = [1] * len(ids)

        padding_length = max_length - len(ids)

        ids.extend(
            [tokenizer.pad_token_id or 0]
            * padding_length
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


def predict_probability(url: str) -> float:
    input_ids, attention_masks = (
        create_url_chunks(url)
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
        )

    # Mean logits across all URL chunks.
    # This is the same aggregation used during evaluation.
    mean_logits = outputs.logits.mean(dim=0)

    probabilities = torch.softmax(
        mean_logits,
        dim=0,
    )

    # Label 1 = phishing
    return float(probabilities[1])


def predict_url(url: str) -> dict[str, Any]:
    probability = predict_probability(url)

    probability = round(
        probability,
        4,
    )

    score = round(
        probability * 100
    )

    classification = classify_probability(
        probability
    )

    indicator = (
        "ML model estimates a "
        f"{score}% phishing probability."
    )

    return {
        "score": score,
        "classification": classification,
        "probability": probability,
        "indicator": indicator,
    }