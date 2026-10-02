from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold


DATASET_PATH = (
    Path(__file__).parent
    / "data"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)

MODEL_PATH = (
    Path(__file__).parent
    / "hostname_model.joblib"
)


def extract_hostname(url: str) -> str:
    try:
        return (
            urlparse(url).hostname or ""
        ).lower()
    except ValueError:
        return ""


df = pd.read_csv(DATASET_PATH)

df = df[["URL", "label"]].dropna()
df["URL"] = df["URL"].astype(str)
df = df.drop_duplicates(subset=["URL"])

# Original dataset:
# 1 = legitimate
# 0 = phishing
df["target"] = (df["label"] == 0).astype(int)

df["hostname"] = df["URL"].apply(
    extract_hostname
)

df = df[df["hostname"] != ""]

X = df["hostname"]
y = df["target"]
groups = df["hostname"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

train_indices, test_indices = next(
    splitter.split(X, y, groups)
)

X_train = X.iloc[train_indices]
X_test = X.iloc[test_indices]

y_train = y.iloc[train_indices]
y_test = y.iloc[test_indices]

print("Dataset shape:")
print(df[["hostname", "target"]].shape)

print(f"Training records: {len(X_train)}")
print(f"Test records: {len(X_test)}")

print()
print("Training hostname character-level model...")

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(2, 5),
    min_df=2,
    max_features=50000,
    sublinear_tf=True,
    lowercase=True,
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

model = LogisticRegression(
    max_iter=300,
    class_weight="balanced",
    solver="liblinear",
    random_state=42,
)

model.fit(
    X_train_vectorized,
    y_train,
)

predictions = model.predict(
    X_test_vectorized
)

print()
print("Classification report:")
print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "legitimate",
            "phishing",
        ],
        digits=4,
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(
        y_test,
        predictions,
    )
)

joblib.dump(
    {
        "model": model,
        "vectorizer": vectorizer,
    },
    MODEL_PATH,
)

print()
print(f"Hostname model saved to: {MODEL_PATH}")