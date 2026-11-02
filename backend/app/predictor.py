"""URL phishing probability from the URLBERT model.

The model is trained with PyTorch (see research/) and exported to
ONNX, so the backend only needs ONNX Runtime and the tokenizers
library, which use far less memory than PyTorch.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime
from tokenizers import Tokenizer


MODEL_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "urlbert_combined_v2"
)

CHUNK_OVERLAP = 16


def load_preprocessing(model_directory: Path) -> dict[str, Any]:
    # Newer models describe their input preprocessing in
    # preprocessing.json, older models have no such file.
    preprocessing_file = model_directory / "preprocessing.json"

    if not preprocessing_file.exists():
        return {}

    return json.loads(
        preprocessing_file.read_text(encoding="utf-8")
    )


PREPROCESSING = load_preprocessing(MODEL_DIRECTORY)

MAX_LENGTH = int(PREPROCESSING.get("max_length", 64))

tokenizer = Tokenizer.from_file(
    str(MODEL_DIRECTORY / "tokenizer.json")
)

CLS_TOKEN_ID = tokenizer.token_to_id("[CLS]")
SEP_TOKEN_ID = tokenizer.token_to_id("[SEP]")
PAD_TOKEN_ID = tokenizer.token_to_id("[PAD]") or 0

session = onnxruntime.InferenceSession(
    str(MODEL_DIRECTORY / "model.onnx"),
    providers=["CPUExecutionProvider"],
)


def preprocess_url(
    url: str,
    preprocessing: dict[str, Any] = PREPROCESSING,
) -> str:
    url = url.strip()

    if preprocessing.get("strip_scheme"):
        for scheme in ("https://", "http://"):
            if url.lower().startswith(scheme):
                return url[len(scheme):]

    return url


def classify_probability(
    probability: float,
) -> str:
    if probability >= 0.70:
        return "Likely phishing"

    if probability >= 0.40:
        return "Suspicious"

    return "Likely safe"


def tokenize_url(url: str) -> list[int]:
    return tokenizer.encode(
        url,
        add_special_tokens=False,
    ).ids


def create_chunks(
    token_ids: list[int],
    cls_token_id: int = CLS_TOKEN_ID,
    sep_token_id: int = SEP_TOKEN_ID,
    pad_token_id: int = PAD_TOKEN_ID,
    max_length: int = MAX_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Split token ids into overlapping chunks of max_length tokens.

    The model accepts at most max_length tokens, longer URLs are
    split with an overlap of CHUNK_OVERLAP tokens. The special token
    ids are parameters so the research scripts can use the same
    chunking with other tokenizers.
    """

    content_length = max_length - 2

    if len(token_ids) <= content_length:
        chunks = [token_ids]
    else:
        step = content_length - CHUNK_OVERLAP
        chunks = []

        for start in range(0, len(token_ids), step):
            chunk = token_ids[start:start + content_length]

            if chunk:
                chunks.append(chunk)

            if start + content_length >= len(token_ids):
                break

    input_ids = np.full(
        (len(chunks), max_length),
        pad_token_id,
        dtype=np.int64,
    )

    attention_mask = np.zeros(
        (len(chunks), max_length),
        dtype=np.int64,
    )

    for row, chunk in enumerate(chunks):
        ids = [cls_token_id, *chunk, sep_token_id]

        input_ids[row, :len(ids)] = ids
        attention_mask[row, :len(ids)] = 1

    return input_ids, attention_mask


def softmax(logits: np.ndarray) -> np.ndarray:
    exponentials = np.exp(logits - logits.max(axis=-1, keepdims=True))

    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def predict_probabilities(
    urls: list[str],
    batch_size: int = 256,
) -> np.ndarray:
    """Phishing probability for each URL.

    The logits of all chunks of a URL are averaged, the same
    aggregation as during training and evaluation.
    """

    probabilities = np.zeros(len(urls))

    for start in range(0, len(urls), batch_size):
        batch = urls[start:start + batch_size]

        chunked = [
            create_chunks(tokenize_url(preprocess_url(url)))
            for url in batch
        ]

        logits = session.run(
            ["logits"],
            {
                "input_ids": np.concatenate([ids for ids, _ in chunked]),
                "attention_mask": np.concatenate([mask for _, mask in chunked]),
            },
        )[0]

        chunk_counts = [len(ids) for ids, _ in chunked]
        boundaries = np.cumsum(chunk_counts)[:-1]

        url_logits = np.stack(
            [
                chunk_logits.mean(axis=0)
                for chunk_logits in np.split(logits, boundaries)
            ]
        )

        # Label 1 = phishing
        probabilities[start:start + len(batch)] = softmax(url_logits)[:, 1]

    return probabilities


def predict_probability(url: str) -> float:
    return float(predict_probabilities([url])[0])


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
