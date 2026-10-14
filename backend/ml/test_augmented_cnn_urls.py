from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ml.benchmark_deep_models import CNN, URLDataset


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "ml" / "cnn_augmented_model.pt"


TEST_URLS = [
    ("https://www.google.com/", 0),
    ("https://mail.google.com/mail/u/0/#inbox", 0),
    ("https://finki.ukim.mk/", 0),
    (
        "https://finki.ukim.mk/studii-2/vidovi-studii/"
        "dodiplomski-studii/",
        0,
    ),
    ("https://www.microsoft.com/", 0),
    ("https://github.com/", 0),
    (
        "https://docs.python.org/3/library/urllib.parse.html",
        0,
    ),
    ("https://www.booking.com/", 0),

    (
        "http://secure-login-verification.example/login",
        1,
    ),
    (
        "http://account-confirmation.example/verify/password",
        1,
    ),
    (
        "http://paypal-secure.example/verify/account",
        1,
    ),
    (
        "http://192.0.2.10/login",
        1,
    ),
]


def main():
    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
    )

    vocabulary = checkpoint["vocabulary"]

    model = CNN(len(vocabulary) + 2)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    urls = [item[0] for item in TEST_URLS]
    expected_labels = [item[1] for item in TEST_URLS]

    dataset = URLDataset(
        urls,
        expected_labels,
        vocabulary,
    )

    loader = DataLoader(
        dataset,
        batch_size=32,
    )

    probabilities = []

    with torch.no_grad():
        for features, _ in loader:
            scores = torch.sigmoid(
                model(features)
            )
            probabilities.extend(scores.tolist())

    print("\nAUGMENTED CNN URL TEST RESULTS")
    print("=" * 70)

    for (url, expected), probability in zip(
        TEST_URLS,
        probabilities,
    ):
        predicted_label = int(probability >= 0.5)

        expected_text = (
            "phishing"
            if expected == 1
            else "legitimate"
        )

        predicted_text = (
            "phishing"
            if predicted_label == 1
            else "legitimate"
        )

        print(f"\nURL: {url}")
        print(f"Expected: {expected_text}")
        print(f"Prediction: {predicted_text}")
        print(f"Phishing probability: {probability:.2%}")


if __name__ == "__main__":
    main()