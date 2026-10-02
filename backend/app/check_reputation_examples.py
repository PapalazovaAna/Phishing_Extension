from app.reputation import check_reputation


TEST_URLS = [
    "https://google.com/",
    "https://www.google.com/",
    "https://mail.google.com/",
    "https://mail.google.com/mail/u/0/#inbox",
    "https://microsoft.com/",
    "https://rnicrosoft.com/",
    "https://booking.com/",
    "https://booking.com/payments",
]


for url in TEST_URLS:
    result = check_reputation(url)

    print("\n" + "=" * 70)
    print(url)
    print(result)