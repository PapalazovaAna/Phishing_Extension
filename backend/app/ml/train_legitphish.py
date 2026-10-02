from pathlib import Path
from urllib.parse import urlparse

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

from features import extract_url_features


DATASET_PATH = (
    Path(__file__).parent
    / "data"
    / "url_features_extracted1.csv"
)

MODEL_PATH = (
    Path(__file__).parent
    / "legitphish_url_model.joblib"
)


def extract_domain(url: str) -> str:
    try:
        return (
            urlparse(url).hostname or ""
        ).lower()
    except ValueError:
        return ""


df = pd.read_csv(DATASET_PATH)

df = df[["URL", "ClassLabel"]].dropna()

df["URL"] = df["URL"].astype(str)
df["ClassLabel"] = df["ClassLabel"].astype(int)

df = df.drop_duplicates(
    subset=["URL"]
)

# LegitPhish:
# 0 = phishing
# 1 = legitimate
#
# Our convention:
# 0 = legitimate
# 1 = phishing
df["target"] = (
    df["ClassLabel"] == 0
).astype(int)

df["domain"] = df["URL"].apply(
    extract_domain
)

df = df[df["domain"] != ""]

feature_rows = [
    extract_url_features(url)
    for url in df["URL"]
]

features = pd.DataFrame(feature_rows)

# HTTPS is handled by the rule-based analyzer.
features = features.drop(
    columns=["has_https"]
)

target = df["target"]
groups = df["domain"]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

train_indices, test_indices = next(
    splitter.split(
        features,
        target,
        groups,
    )
)

X_train = features.iloc[train_indices]
X_test = features.iloc[test_indices]

y_train = target.iloc[train_indices]
y_test = target.iloc[test_indices]

print("Dataset shape:")
print(df[["URL", "target", "domain"]].shape)

print(f"Training records: {len(X_train)}")
print(f"Test records: {len(X_test)}")

print()
print("Class distribution:")
print(
    df["target"].map(
        {
            0: "legitimate",
            1: "phishing",
        }
    ).value_counts()
)

model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

print()
print("Training LegitPhish Random Forest...")

model.fit(
    X_train,
    y_train,
)

predictions = model.predict(
    X_test
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
        "feature_names": list(
            features.columns
        ),
        "dataset": "LegitPhish",
    },
    MODEL_PATH,
)

print()
print(f"Model saved to: {MODEL_PATH}")