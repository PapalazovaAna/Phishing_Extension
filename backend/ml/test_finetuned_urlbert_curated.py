from pathlib import Path
import argparse

import torch
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATHS = {
    "vonpower": (
        BASE_DIR
        / "ml"
        / "urlbert_finetuned_vonpower"
    ),
    "ealvaradob": (
        BASE_DIR
        / "ml"
        / "urlbert_finetuned_ealvaradob"
    ),
}

CHUNK_OVERLAP = 16

TEST_URLS = {
    "Google": (
        "https://www.google.com/",
        "legitimate",
    ),
    "Gmail": (
        "https://mail.google.com/mail/u/0/#inbox",
        "legitimate",
    ),
    "FINKI": (
        "https://finki.ukim.mk/",
        "legitimate",
    ),
    "FINKI studies": (
        "https://finki.ukim.mk/"
        "studii-2/vidovi-studii/"
        "dodiplomski-studii/",
        "legitimate",
    ),
    "Microsoft": (
        "https://www.microsoft.com/",
        "legitimate",
    ),
    "GitHub": (
        "https://github.com/",
        "legitimate",
    ),
    "Python documentation": (
        "https://docs.python.org/3/library/urllib.parse.html",
        "legitimate",
    ),
    "Booking": (
        "https://www.booking.com/",
        "legitimate",
    ),
    "Typosquatting Microsoft": (
        "https://rnicrosoft.com/",
        "phishing",
    ),
    "Suspicious login": (
        "http://secure-login-verification-example.com/login",
        "phishing",
    ),
    "Suspicious account": (
        "http://account-confirmation-example.com/verify/password",
        "phishing",
    ),
    "Suspicious payment": (
        "http://paypal-secure-example.com/verify/account",
        "phishing",
    ),
    "IP address URL": (
        "http://192.168.1.25/login",
        "phishing",
    ),
}


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

    input_ids = []
    attention_masks = []

    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id or 0

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


def predict_probability(
    url,
    model,
    tokenizer,
    model_max_length,
):
    input_ids, attention_masks = create_chunks(
        url,
        tokenizer,
        model_max_length,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_masks,
        )

    mean_logits = outputs.logits.mean(dim=0)

    probabilities = torch.softmax(
        mean_logits,
        dim=0,
    )

    return float(probabilities[1])


def evaluate_model(model_name):
    model_path = MODEL_PATHS[model_name]

    print(f"\nLoading fine-tuned URLBERT: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(model_path)
    )

    model.eval()

    model_max_length = int(
        model.config.max_position_embeddings
    )

    predictions = []
    expected_labels = []

    print("\n" + "=" * 80)
    print(
        f"URLBERT FINE-TUNED ON {model_name.upper()}"
    )
    print("=" * 80)

    for name, (url, expected) in TEST_URLS.items():
        probability = predict_probability(
            url,
            model,
            tokenizer,
            model_max_length,
        )

        prediction = (
            "phishing"
            if probability >= 0.5
            else "legitimate"
        )

        predictions.append(prediction)
        expected_labels.append(expected)

        print(f"\n{name}")
        print(f"URL: {url}")
        print(f"Expected: {expected}")
        print(f"Prediction: {prediction}")
        print(
            "Phishing probability: "
            f"{probability * 100:.2f}%"
        )

    accuracy = accuracy_score(
        expected_labels,
        predictions,
    )

    legitimate_total = sum(
        label == "legitimate"
        for label in expected_labels
    )

    phishing_total = sum(
        label == "phishing"
        for label in expected_labels
    )

    legitimate_correct = sum(
        expected == "legitimate"
        and prediction == "legitimate"
        for expected, prediction in zip(
            expected_labels,
            predictions,
        )
    )

    phishing_correct = sum(
        expected == "phishing"
        and prediction == "phishing"
        for expected, prediction in zip(
            expected_labels,
            predictions,
        )
    )

    print("\n" + "-" * 80)
    print(
        f"Total accuracy: "
        f"{accuracy * 100:.2f}%"
    )
    print(
        f"Legitimate accuracy: "
        f"{legitimate_correct}/{legitimate_total} "
        f"({legitimate_correct / legitimate_total * 100:.2f}%)"
    )
    print(
        f"Phishing accuracy: "
        f"{phishing_correct}/{phishing_total} "
        f"({phishing_correct / phishing_total * 100:.2f}%)"
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "vonpower",
            "ealvaradob",
            "both",
        ],
        default="both",
    )

    args = parser.parse_args()

    if args.model == "both":
        evaluate_model("vonpower")
        evaluate_model("ealvaradob")
    else:
        evaluate_model(args.model)


if __name__ == "__main__":
    main()