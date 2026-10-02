from pathlib import Path

import joblib
import pandas as pd

from features import extract_url_features


MODEL_PATH = (
    Path(__file__).parent
    / "legitphish_url_model.joblib"
)


artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
feature_names = artifact["feature_names"]


test_urls = {
    "legitimate_google": "https://www.google.com/",
    "legitimate_booking": "https://booking.com/payments",
    "legitimate_microsoft": "https://www.microsoft.com/",
    "legitimate_gmail": "https://mail.google.com/mail/u/0/#inbox",
    "legitimate_github": "https://github.com/",
    "typosquatting_microsoft": "https://rnicrosoft.com/",
    "suspicious_login": (
        "http://secure-login-verification-example.com/login"
    ),
    "suspicious_account": (
        "http://account-confirmation-example.com/verify/password"
    ),
}


for name, url in test_urls.items():
    extracted_features = extract_url_features(url)

    row = {
        feature_name: extracted_features.get(
            feature_name,
            0,
        )
        for feature_name in feature_names
    }

    features = pd.DataFrame(
        [row],
        columns=feature_names,
    )

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    phishing_index = list(
        model.classes_
    ).index(1)

    phishing_probability = (
        probabilities[phishing_index] * 100
    )

    classification = (
        "phishing"
        if prediction == 1
        else "legitimate"
    )

    print()
    print(name)
    print(f"URL: {url}")
    print(f"Prediction: {classification}")
    print(
        "Phishing probability: "
        f"{phishing_probability:.2f}%"
    )