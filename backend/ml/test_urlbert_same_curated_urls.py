import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_NAME = (
    "CrabInHoney/"
    "urlbert-tiny-v4-phishing-classifier"
)

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

    all_input_ids = []
    all_attention_masks = []

    pad_token_id = tokenizer.pad_token_id or 0
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id

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
        all_attention_masks.append(attention_mask)

    return (
        torch.tensor(
            all_input_ids,
            dtype=torch.long,
        ),
        torch.tensor(
            all_attention_masks,
            dtype=torch.long,
        ),
    )


def predict_probability(
    url,
    model,
    tokenizer,
    model_max_length,
):
    input_ids, attention_mask = create_chunks(
        url,
        tokenizer,
        model_max_length,
    )

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # Mean logits across all chunks
    mean_logits = outputs.logits.mean(dim=0)

    probabilities = torch.softmax(
        mean_logits,
        dim=0,
    )

    # URLBERT label 1 = phishing
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

    model_max_length = int(
        model.config.max_position_embeddings
    )

    print(
        f"Model maximum length: "
        f"{model_max_length}"
    )

    correct = 0
    legitimate_correct = 0
    legitimate_total = 0
    phishing_correct = 0
    phishing_total = 0

    print("\nURLBERT CURATED URL TEST")
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
            if probability >= 0.50
            else "legitimate"
        )

        is_correct = prediction == expected

        if is_correct:
            correct += 1

        if expected == "legitimate":
            legitimate_total += 1

            if is_correct:
                legitimate_correct += 1

        else:
            phishing_total += 1

            if is_correct:
                phishing_correct += 1

        print(f"\n{name}")
        print(f"URL: {url}")
        print(f"Expected: {expected}")
        print(f"Prediction: {prediction}")
        print(
            "Phishing probability: "
            f"{probability * 100:.2f}%"
        )

    total = len(TEST_URLS)

    print("\n" + "=" * 80)
    print(
        f"Total accuracy: "
        f"{correct}/{total} "
        f"({correct / total * 100:.2f}%)"
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


if __name__ == "__main__":
    main()