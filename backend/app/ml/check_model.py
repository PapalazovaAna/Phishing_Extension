import pandas as pd

from predictor import predict_url


DATASET_PATH = "app/ml/data/PhiUSIIL_Phishing_URL_Dataset.csv"


df = pd.read_csv(DATASET_PATH)

legitimate_urls = (
    df[df["label"] == 1]["URL"]
    .dropna()
    .sample(10, random_state=42)
)

phishing_urls = (
    df[df["label"] == 0]["URL"]
    .dropna()
    .sample(10, random_state=42)
)

print("LEGITIMATE URLS")
print("=" * 60)

for url in legitimate_urls:
    result = predict_url(url)
    print(f"{result['score']:>3}% | {url}")


print()
print("PHISHING URLS")
print("=" * 60)

for url in phishing_urls:
    result = predict_url(url)
    print(f"{result['score']:>3}% | {url}")