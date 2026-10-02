from pathlib import Path
from urllib.parse import urlparse

import joblib


MODEL_PATH = (
    Path(__file__).parent
    / "hostname_model.joblib"
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


def extract_hostname(url: str) -> str:
    return (
        urlparse(url).hostname or ""
    ).lower()


names = list(test_urls.keys())
urls = list(test_urls.values())

hostnames = [
    extract_hostname(url)
    for url in urls
]

vectorized_hosts = vectorizer.transform(hostnames)

predictions = model.predict(vectorized_hosts)
probabilities = model.predict_proba(vectorized_hosts)

phishing_index = list(
    model.classes_
).index(1)


for name, url, prediction, probability in zip(
    names,
    urls,
    predictions,
    probabilities,
):
    phishing_probability = (
        probability[phishing_index] * 100
    )

    classification = (
        "phishing"
        if prediction == 1
        else "legitimate"
    )

    print()
    print(name)
    print(f"URL: {url}")
    print(f"Hostname: {extract_hostname(url)}")
    print(f"Prediction: {classification}")
    print(
        "Phishing probability: "
        f"{phishing_probability:.2f}%"
    )