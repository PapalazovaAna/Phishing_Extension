from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


MODEL_ID = "CrabInHoney/urlbert-tiny-v6"
HEAD_NAME = "phiusiil_phishing_url"

MODEL_DIR = (
    Path(__file__).resolve().parent
    / "models"
    / "urlbert-tiny-v6"
)


TEST_URLS = {
    "Google": "https://www.google.com/",
    "Booking": "https://booking.com/payments",
    "Booking query": (
        "https://www.booking.com/"
        "?chal_t=1790763960679&force_referer="
    ),
    "Long Booking URL": (
        "https://www.booking.com/searchresults.html?"
        "si=ai%2Cco%2Cci%2Cre%2Cla%2Cdi;"
        "dest_type=hotel;"
        "dest_id=8767025;"
        "ifl=1;"
        "aid=304142;"
        "label=gen173nr-10CBcoggI46AdIM1gEaJMBiAEBmAEzuAEXyAEM2AED6AEB-"
        "AEBiAIBqAIBuAKQn9zVBsACAdICJGFiZjBjOGE2LTNhZTMtNDkyNi1iY2Y5LWI0ZTc5OTI4NzQ0YdgCAeACAQ"
    ),
    "FINKI homepage": "https://finki.ukim.mk/",
    "FINKI undergraduate studies": (
        "https://finki.ukim.mk/"
        "studii-2/vidovi-studii/dodiplomski-studii/"
    ),
    "FINKI masters quotas": (
        "https://finki.ukim.mk/"
        "upisi/magisterski-studii/kvoti/"
    ),
    "Microsoft": "https://www.microsoft.com/",
    "Gmail": "https://mail.google.com/mail/u/0/#inbox",
    "Gmail with query": (
        "https://mail.google.com/mail/u/0/"
        "?tab=rm&ogbl#inbox"
    ),
    "GitHub": "https://github.com/",
    "Typosquatting Microsoft": "https://rnicrosoft.com/",
    "Suspicious login": (
        "http://secure-login-verification.example/login"
    ),
    "Suspicious account": (
        "http://account-confirmation.example/verify/password"
    ),
    "IP address URL": "http://192.0.2.10/login",
}


def download_model() -> None:
    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        print(f"Model already exists at: {MODEL_DIR}")
        return

    print("Downloading URLBERT v6 model...")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(MODEL_DIR),
    )

    print(f"Model saved to: {MODEL_DIR}")


def find_phishing_label(model) -> int:
    id_to_label = model.config.id2label

    for label_id, label_name in id_to_label.items():
        normalized_label = str(label_name).lower()

        if any(
            keyword in normalized_label
            for keyword in [
                "phishing",
                "phish",
                "malicious",
                "unsafe",
            ]
        ):
            return int(label_id)

    print(
        "Could not identify phishing label from model config."
    )
    print(f"Available labels: {id_to_label}")
    print("Assuming label 1 is phishing.")

    return 1


def predict_url(
    url: str,
    tokenizer,
    model,
    phishing_label: int,
) -> tuple[str, float]:
    inputs = tokenizer(
        url,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    phishing_probability = float(
        probabilities[phishing_label]
    )

    prediction_label = int(
        torch.argmax(probabilities).item()
    )

    prediction = (
        "phishing"
        if prediction_label == phishing_label
        else "legitimate"
    )

    return prediction, phishing_probability


def main() -> None:
    download_model()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
    )

    print("Loading PhiUSIIL classification head...")
    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR),
        subfolder=f"heads/{HEAD_NAME}",
        trust_remote_code=True,
    )

    model.eval()

    phishing_label = find_phishing_label(model)

    print("=" * 70)
    print("URLBERT V6 EVALUATION")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Classification head: {HEAD_NAME}")
    print(f"Labels: {model.config.id2label}")
    print(f"Phishing label index: {phishing_label}")
    print("=" * 70)

    for name, url in TEST_URLS.items():
        prediction, probability = predict_url(
            url=url,
            tokenizer=tokenizer,
            model=model,
            phishing_label=phishing_label,
        )

        print()
        print(name)
        print(f"URL: {url}")
        print(f"Prediction: {prediction}")
        print(
            "Phishing probability: "
            f"{probability * 100:.2f}%"
        )


if __name__ == "__main__":
    main()