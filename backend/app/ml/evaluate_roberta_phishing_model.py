import torch

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_NAME = (
    "nhellyercreek/url-phishing-classifier"
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
    "Long Booking URL": (
        "https://www.booking.com/"
        "searchresults.html?"
        "si=ai%2Cco%2Cci%2Cre%2Cla%2Cdi;"
        "dest_type=hotel;"
        "dest_id=8767025;"
        "ifl=1;"
        "aid=304142;"
        "label=gen173nr-10CBcoggI46AdIM1gEaJMBiAEBmAEzuAEXyAEM2AED6AEB-AEBiAIBqAIBuAKQn9zVBsACAdICJGFiZjBjOGE2LTNhZTMtNDkyNi1iY2Y5LWI0ZTc5OTI4NzQ0YdgCAeACAQ"
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
        "http://192.0.2.10/login"
    ),
}


def predict_probability(
    url: str,
    tokenizer,
    model,
) -> float:
    model_max_length = int(
        model.config.max_position_embeddings
    )

    max_length = min(
        256,
        model_max_length,
    )

    inputs = tokenizer(
        url,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=1,
    )

    # LABEL 1 = phishing
    return float(
        probabilities[0][1]
    )


def main() -> None:
    print("Downloading/loading RoBERTa model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(MODEL_NAME)
    )

    model.eval()

    print("=" * 70)
    print("ROBERTA PHISHING MODEL")
    print("=" * 70)
    print(f"Model: {MODEL_NAME}")
    print(
        "Labels: LABEL_0 = safe, "
        "LABEL_1 = phishing"
    )

    for name, url in TEST_URLS.items():
        probability = predict_probability(
            url=url,
            tokenizer=tokenizer,
            model=model,
        )

        classification = (
            "phishing"
            if probability >= 0.50
            else "legitimate"
        )

        print(f"\n{name}")
        print(f"URL: {url}")
        print(
            f"Prediction: {classification}"
        )
        print(
            "Phishing probability: "
            f"{probability * 100:.2f}%"
        )


if __name__ == "__main__":
    main()