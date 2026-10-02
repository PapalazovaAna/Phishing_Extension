from pathlib import Path

import joblib


MODEL_PATH = (
    Path(__file__).parent
    / "char_model.joblib"
)


artifact = joblib.load(MODEL_PATH)

model = artifact["model"]
vectorizer = artifact["vectorizer"]


test_urls = {
    "legitimate_google": "https://www.google.com/",
    "legitimate_booking": "https://booking.com/payments",
    "legitimate_microsoft": "https://www.microsoft.com/",
    "legitimate_gmail": "https://mail.google.com/mail/u/0/#inbox",
    "typosquatting_microsoft": "https://rnicrosoft.com/",
    "suspicious_login": (
        "http://secure-login-verification-example.com/login"
    ),
    "suspicious_account": (
        "http://account-confirmation-example.com/verify/password"
    ),
}


urls = list(test_urls.values())
vectorized_urls = vectorizer.transform(urls)

predictions = model.predict(vectorized_urls)
probabilities = model.predict_proba(vectorized_urls)


for (name, url), prediction, probability in zip(
    test_urls.items(),
    predictions,
    probabilities,
):
    phishing_index = list(model.classes_).index(1)
    phishing_probability = probability[phishing_index] * 100

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