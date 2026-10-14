from app.ml.predictor import predict_url


TEST_URLS = [
    "https://www.google.com/",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://finki.ukim.mk/",
    "https://finki.ukim.mk/studii-2/vidovi-studii/dodiplomski-studii/",
    "https://www.microsoft.com/",
    "https://github.com/",
    "https://docs.python.org/3/library/urllib.parse.html",
    "https://www.booking.com/",
    "https://rnicrosoft.com/",
    "http://secure-login-verification-example.com/login",
    "http://account-confirmation-example.com/verify/password",
    "http://paypal-secure-example.com/verify/account",
    "http://192.168.1.25/login",
]


def main():
    for url in TEST_URLS:
        result = predict_url(url)

        print("\nURL:", url)
        print(
            "Classification:",
            result["classification"],
        )
        print(
            "Phishing probability:",
            f"{result['probability'] * 100:.2f}%",
        )


if __name__ == "__main__":
    main()