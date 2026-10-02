from pathlib import Path
import sys

import joblib
import pandas as pd


CURRENT_DIRECTORY = Path(__file__).resolve().parent

if str(CURRENT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIRECTORY))

from legitphish_features import (
    extract_legitphish_features,
)


MODEL_PATH = (
    CURRENT_DIRECTORY
    / "combined_url_model.joblib"
)


TEST_URLS = {
    "Google": (
        "https://www.google.com/"
    ),
    "Booking": (
        "https://booking.com/payments"
    ),
    "Booking with query": (
        "https://www.booking.com/"
        "?chal_t=1790381880757"
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
    "Suspicious payment URL": (
        "http://paypal-secure-confirmation.example/"
        "verify/account"
    ),
}


def main() -> None:
    artifact = joblib.load(MODEL_PATH)

    model = artifact["model"]
    feature_names = artifact["feature_names"]

    print("=" * 70)
    print("COMBINED MODEL - EXTERNAL URL EVALUATION")
    print("=" * 70)
    print(f"Model: {artifact['model_name']}")
    print(
        f"Training phishing F1: "
        f"{artifact['phishing_f1']:.4f}"
    )

    for name, url in TEST_URLS.items():
        extracted_features = (
            extract_legitphish_features(url)
        )

        row = pd.DataFrame(
            [[
                extracted_features[feature]
                for feature in feature_names
            ]],
            columns=feature_names,
        )

        probability = model.predict_proba(row)[0][1]
        prediction = int(probability >= 0.50)

        classification = (
            "phishing"
            if prediction == 1
            else "legitimate"
        )

        print("\n" + name)
        print(f"URL: {url}")
        print(
            f"Prediction: {classification}"
        )
        print(
            f"Phishing probability: "
            f"{probability * 100:.2f}%"
        )


if __name__ == "__main__":
    main()