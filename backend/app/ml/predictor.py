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
    / "urlbert-tiny-v4-phishing-classifier"
)


tokenizer = AutoTokenizer.from_pretrained(
    str(MODEL_DIRECTORY),
    local_files_only=True,
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(MODEL_DIRECTORY),
    local_files_only=True,
)

model.eval()


def classify_probability(
    probability: float,
) -> str:
    if probability >= 0.70:
        return "Likely phishing"

    if probability >= 0.40:
        return "Suspicious"

    return "Likely safe"


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

def predict_probability(url: str) -> float:
    max_length = int(
        model.config.max_position_embeddings
    )

    encoded = tokenizer(
        url,
        truncation=True,
        max_length=max_length,
        stride=16,
        padding=True,
        return_overflowing_tokens=True,
        return_tensors="pt",
    )

    model_inputs = {
        key: value
        for key, value in encoded.items()
        if key in {
            "input_ids",
            "attention_mask",
            "token_type_ids",
        }
    }

    with torch.no_grad():
        outputs = model(**model_inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1,
    )

    chunk_probabilities = probabilities[:, 1]

    # The most suspicious chunk determines the ML score.
    return float(
        chunk_probabilities.max()
    )