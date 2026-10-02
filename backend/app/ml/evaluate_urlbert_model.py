import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_NAME = (
    "CrabInHoney/"
    "urlbert-tiny-v4-phishing-classifier"
)


TEST_URLS = {
    "Google": (
        "https://www.google.com/"
    ),
    "Booking": (
        "https://booking.com/payments"
    ),
    "Booking query": (
        "https://www.booking.com/"
        "?chal_t=1790763960679"
        "&force_referer="
    ),
    "Microsoft": (
        "https://www.microsoft.com/"
    ),
    "Gmail": (
        "https://mail.google.com/"
        "mail/u/0/#inbox"
    ),
    "GitHub": (
        "https://github.com/"
    ),
    "Typosquatting Microsoft": (
        "https://rnicrosoft.com/"
    ),
    "Suspicious login": (
        "http://secure-login-verification-example.com/"
        "login"
    ),
    "Suspicious account": (
        "http://account-confirmation-example.com/"
        "verify/password"
    ),
    "IP address URL": (
        "http://192.168.1.25/login"
    ),
}


def main() -> None:
    print("Downloading/loading URLBERT model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(MODEL_NAME)
    )

    model.eval()

    urls = list(TEST_URLS.values())

    inputs = tokenizer(
        urls,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1,
    )

    label_mapping = model.config.id2label

    print("=" * 70)
    print("URLBERT HUGGING FACE MODEL")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(f"Labels: {label_mapping}")

    for index, (name, url) in enumerate(
        TEST_URLS.items()
    ):
        phishing_probability = float(
            probabilities[index][1]
        )

        classification = (
            "phishing"
            if phishing_probability >= 0.50
            else "legitimate"
        )

        print(f"\n{name}")
        print(f"URL: {url}")
        print(
            f"Prediction: {classification}"
        )
        print(
            "Phishing probability: "
            f"{phishing_probability * 100:.2f}%"
        )


if __name__ == "__main__":
    main()