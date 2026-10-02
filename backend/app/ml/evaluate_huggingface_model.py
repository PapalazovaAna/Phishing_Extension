from pathlib import Path

import numpy as np
import onnxruntime
from huggingface_hub import hf_hub_download


MODEL_REPOSITORY = (
    "pirocheto/phishing-url-detection"
)

MODEL_FILENAME = "model.onnx"


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
    "Booking search": (
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
        "http://192.168.1.25/login"
    ),
}


def main() -> None:
    print("Downloading/loading Hugging Face model...")

    model_path = hf_hub_download(
        repo_id=MODEL_REPOSITORY,
        filename=MODEL_FILENAME,
    )

    session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            "CPUExecutionProvider"
        ],
    )

    urls = list(TEST_URLS.values())

    inputs = np.array(
        urls,
        dtype=str,
    )

    outputs = session.run(
        None,
        {
            "inputs": inputs,
        },
    )

    probabilities = outputs[1]

    print("=" * 70)
    print("HUGGING FACE MODEL EVALUATION")
    print("=" * 70)
    print(f"Model: {MODEL_REPOSITORY}")
    print(f"Model file: {model_path}")

    for (
        (name, url),
        probability,
    ) in zip(
        TEST_URLS.items(),
        probabilities,
    ):
        phishing_probability = float(
            probability[1]
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