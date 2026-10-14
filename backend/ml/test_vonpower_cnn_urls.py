from pathlib import Path

import torch

from ml.train_vonpower_cnn import URLCNN


MODEL_PATH = (
    Path(__file__).resolve().parents[1]
    / "ml"
    / "vonpower_cnn_model.pt"
)


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


def encode_url(url, char_to_int, sequence_length):
    unknown_token = char_to_int.get("�", 0)

    encoded = [
        char_to_int.get(character, unknown_token)
        for character in url
    ]

    encoded = encoded[:sequence_length]

    if len(encoded) < sequence_length:
        encoded.extend(
            [0] * (sequence_length - len(encoded))
        )

    return torch.tensor(
        [encoded],
        dtype=torch.long,
    )


def main():
    print("Loading vonpower CNN...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    char_to_int = checkpoint["char_to_int"]
    sequence_length = checkpoint["sequence_length"]
    vocabulary_size = checkpoint["vocabulary_size"]

    model = URLCNN(
        vocabulary_size=vocabulary_size
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("=" * 80)
    print("VONPOWER CNN URL TEST")
    print("=" * 80)

    correct = 0

    for name, (url, expected) in TEST_URLS.items():
        inputs = encode_url(
            url,
            char_to_int,
            sequence_length,
        )

        with torch.no_grad():
            logits = model(inputs)
            probability = torch.sigmoid(logits).item()

        prediction = (
            "phishing"
            if probability >= 0.5
            else "legitimate"
        )

        if prediction == expected:
            correct += 1

        print(f"\n{name}")
        print(f"URL: {url}")
        print(f"Expected: {expected}")
        print(f"Prediction: {prediction}")
        print(
            "Phishing probability: "
            f"{probability * 100:.2f}%"
        )

    accuracy = correct / len(TEST_URLS)

    print("\n" + "=" * 80)
    print(
        f"Curated accuracy: "
        f"{correct}/{len(TEST_URLS)} "
        f"({accuracy * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()