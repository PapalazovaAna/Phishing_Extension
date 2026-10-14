from pathlib import Path

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download


MODEL_ID = "pirocheto/phishing-url-detection"
MODEL_FILENAME = "model.onnx"

MODEL_DIR = (
    Path(__file__).resolve().parent
    / "models"
    / "pirocheto"
)


TEST_GROUPS = {
    "LEGITIMATE STATIC URLS": [
        "https://www.google.com/",
        "https://www.microsoft.com/",
        "https://github.com/",
        "https://www.wikipedia.org/",
        "https://www.python.org/",
        "https://pypi.org/",
    ],
    "LEGITIMATE DYNAMIC URLS": [
        "https://mail.google.com/mail/u/0/#inbox",
        "https://mail.google.com/mail/u/0/?tab=rm&ogbl#inbox",
        "https://www.google.com/search?q=python",
        "https://www.booking.com/payments",
        "https://www.booking.com/?chal_t=1790763960679&force_referer=",
        "https://www.booking.com/searchresults.html?dest_type=hotel&dest_id=8767025",
        "https://finki.ukim.mk/",
        "https://finki.ukim.mk/studii-2/vidovi-studii/dodiplomski-studii/",
        "https://finki.ukim.mk/upisi/magisterski-studii/kvoti/",
        "https://github.com/python/cpython/issues",
        "https://stackoverflow.com/questions/tagged/python",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://docs.python.org/3/library/urllib.parse.html",
    ],
    "SYNTHETIC PHISHING URLS": [
        "http://secure-login-verification.example/login",
        "http://account-confirmation.example/verify/password",
        "http://paypal-secure.example/verify/account",
        "http://192.0.2.10/login",
        "http://example.invalid/user/login/verify/password",
        "http://login-update-security.example/account",
        "http://secure.example/login?redirect=password",
        "http://xn--pple-43d.example/login",
        "http://user@example.invalid/login",
    ],
}


def download_model() -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / MODEL_FILENAME

    if model_path.exists():
        print(f"Model already exists: {model_path}")
        return model_path

    print("Downloading pirocheto ONNX model...")

    downloaded_path = hf_hub_download(
        repo_id=MODEL_ID,
        filename=MODEL_FILENAME,
        local_dir=str(MODEL_DIR),
    )

    print(f"Model saved to: {downloaded_path}")

    return Path(downloaded_path)


def predict_probability(
    session: onnxruntime.InferenceSession,
    url: str,
) -> float:
    input_name = session.get_inputs()[0].name

    inputs = np.array(
        [url],
        dtype=str,
    )

    outputs = session.run(
        None,
        {
            input_name: inputs
        },
    )

    # The model returns probabilities as the second output.
    probabilities = outputs[1]

    # According to the model's usage example,
    # class index 1 represents phishing.
    return float(probabilities[0][1])


def main() -> None:
    model_path = download_model()

    session = onnxruntime.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )

    print("=" * 70)
    print("PIROCHETO MODEL - EXTENDED EVALUATION")
    print("=" * 70)
    print(f"Model: {MODEL_ID}")
    print(f"Model path: {model_path}")
    print("=" * 70)

    total_legitimate = 0
    legitimate_false_positives = 0

    total_phishing = 0
    phishing_true_positives = 0

    for group_name, urls in TEST_GROUPS.items():
        print()
        print(group_name)
        print("-" * 70)

        for url in urls:
            probability = predict_probability(
                session=session,
                url=url,
            )

            prediction = (
                "phishing"
                if probability >= 0.50
                else "legitimate"
            )

            print(
                f"{prediction:12} "
                f"{probability * 100:6.2f}% | "
                f"{url}"
            )

            if group_name.startswith("LEGITIMATE"):
                total_legitimate += 1

                if probability >= 0.50:
                    legitimate_false_positives += 1

            else:
                total_phishing += 1

                if probability >= 0.50:
                    phishing_true_positives += 1

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(
        "Legitimate URLs classified as phishing: "
        f"{legitimate_false_positives}/{total_legitimate}"
    )

    print(
        "Synthetic phishing URLs classified as phishing: "
        f"{phishing_true_positives}/{total_phishing}"
    )


if __name__ == "__main__":
    main()