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
        "https://www.google.com/",
        0,
    ),
    "Gmail": (
        "https://mail.google.com/mail/u/0/#inbox",
        0,
    ),
    "FINKI": (
        "https://finki.ukim.mk/",
        0,
    ),
    "FINKI undergraduate studies": (
        "https://finki.ukim.mk/studii-2/vidovi-studii/"
        "dodiplomski-studii/",
        0,
    ),
    "Microsoft": (
        "https://www.microsoft.com/",
        0,
    ),
    "GitHub": (
        "https://github.com/",
        0,
    ),
    "Python documentation": (
        "https://docs.python.org/3/library/urllib.parse.html",
        0,
    ),
    "Booking": (
        "https://www.booking.com/",
        0,
    ),
    "Booking query": (
        "https://www.booking.com/"
        "?chal_t=1790763960679"
        "&force_referer=",
        0,
    ),
    "Typosquatting Microsoft": (
        "https://rnicrosoft.com/",
        1,
    ),
    "Suspicious login": (
        "http://secure-login-verification-example.com/login",
        1,
    ),
    "Suspicious account": (
        "http://account-confirmation-example.com/"
        "verify/password",
        1,
    ),
    "Suspicious payment": (
        "http://paypal-secure-example.com/"
        "verify/account",
        1,
    ),
    "IP address URL": (
        "http://192.168.1.25/login",
        1,
    ),
}


def find_phishing_label_index(model) -> int:
    labels = {
        int(index): str(label).lower()
        for index, label in model.config.id2label.items()
    }

    print(f"Model labels: {labels}")

    for index, label in labels.items():
        if (
            "phish" in label
            or "malicious" in label
            or "attack" in label
        ):
            return index

    print(
        "Could not identify phishing label automatically. "
        "Assuming index 1."
    )

    return 1


def main() -> None:
    print("Downloading/loading URLBERT model...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    model.eval()

    phishing_index = find_phishing_label_index(model)

    names = list(TEST_URLS.keys())
    urls = [
        TEST_URLS[name][0]
        for name in names
    ]
    expected_labels = [
        TEST_URLS[name][1]
        for name in names
    ]

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

    predictions = []
    phishing_probabilities = []

    print("=" * 80)
    print("URLBERT v4 EVALUATION")
    print("=" * 80)
    print(f"Model: {MODEL_NAME}")

    for index, name in enumerate(names):
        phishing_probability = float(
            probabilities[index][phishing_index]
        )

        prediction = int(
            phishing_probability >= 0.50
        )

        predictions.append(prediction)
        phishing_probabilities.append(
            phishing_probability
        )

        expected = expected_labels[index]

        expected_text = (
            "phishing"
            if expected == 1
            else "legitimate"
        )

        prediction_text = (
            "phishing"
            if prediction == 1
            else "legitimate"
        )

        print(f"\n{name}")
        print(f"URL: {urls[index]}")
        print(f"Expected: {expected_text}")
        print(f"Prediction: {prediction_text}")
        print(
            "Phishing probability: "
            f"{phishing_probability:.2%}"
        )

    correct = sum(
        prediction == expected
        for prediction, expected in zip(
            predictions,
            expected_labels,
        )
    )

    legitimate_indices = [
        index
        for index, label in enumerate(expected_labels)
        if label == 0
    ]

    phishing_indices = [
        index
        for index, label in enumerate(expected_labels)
        if label == 1
    ]

    legitimate_correct = sum(
        predictions[index] == 0
        for index in legitimate_indices
    )

    phishing_correct = sum(
        predictions[index] == 1
        for index in phishing_indices
    )

    print("\nSUMMARY")
    print("=" * 80)
    print(
        f"Accuracy: "
        f"{correct / len(expected_labels):.2%}"
    )
    print(
        "Legitimate recall: "
        f"{legitimate_correct / len(legitimate_indices):.2%}"
    )
    print(
        "Phishing recall: "
        f"{phishing_correct / len(phishing_indices):.2%}"
    )


if __name__ == "__main__":
    main()