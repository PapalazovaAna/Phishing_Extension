from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_sample_weight

from features import extract_url_features


DATASET_PATH = (
    Path(__file__).parent
    / "data"
    / "PhiUSIIL_Phishing_URL_Dataset.csv"
)


def extract_domain(url: str) -> str:
    try:
        return (
            urlparse(url).hostname or ""
        ).lower()
    except ValueError:
        return ""


def phishing_probability(model, features) -> float:
    probabilities = model.predict_proba(features)[0]
    classes = list(model.classes_)
    phishing_index = classes.index(1)

    return float(probabilities[phishing_index])


df = pd.read_csv(DATASET_PATH)

df = df[["URL", "label"]].dropna()
df["URL"] = df["URL"].astype(str)
df = df.drop_duplicates(subset=["URL"])

# Original dataset:
# 1 = legitimate
# 0 = phishing
#
# Our convention:
# 0 = legitimate
# 1 = phishing
df["target"] = (df["label"] == 0).astype(int)
df["domain"] = df["URL"].apply(extract_domain)

feature_rows = [
    extract_url_features(url)
    for url in df["URL"]
]

features = pd.DataFrame(feature_rows)

# HTTPS is handled by the rule-based URL analyzer.
# It is not used by the ML models.
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
    splitter.split(features, target, groups)
)

X_train = features.iloc[train_indices]
X_test = features.iloc[test_indices]

y_train = target.iloc[train_indices]
y_test = target.iloc[test_indices]

sample_weights = compute_sample_weight(
    class_weight="balanced",
    y=y_train,
)


models = {
    "Random Forest - current": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "Random Forest - tuned": RandomForestClassifier(
        n_estimators=400,
        max_depth=25,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=42,
    ),
}


external_urls = {
    "Google": "https://www.google.com/",
    "Booking payments": "https://booking.com/payments",
    "Microsoft": "https://www.microsoft.com/",
    "Gmail": "https://mail.google.com/mail/u/0/#inbox",
    "Typosquatting Microsoft": "https://rnicrosoft.com/",
    "Suspicious login": (
        "http://secure-login-verification-example.com/login"
    ),
}


for model_name, model in models.items():
    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    if model_name == "HistGradientBoosting":
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
        )
    else:
        model.fit(
            X_train,
            y_train,
            sample_weight=sample_weights,
        )

    predictions = model.predict(X_test)

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

    print(
        "Accuracy:",
        round(
            accuracy_score(
                y_test,
                predictions,
            ),
            4,
        ),
    )

    print()
    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    print()
    print("External URL evaluation:")

    for name, url in external_urls.items():
        url_features = pd.DataFrame(
            [
                extract_url_features(url)
            ]
        ).drop(
            columns=["has_https"]
        )

        prediction = model.predict(
            url_features
        )[0]

        probability = (
            phishing_probability(
                model,
                url_features,
            )
            * 100
        )

        classification = (
            "phishing"
            if prediction == 1
            else "legitimate"
        )

        print(
            f"{name:<28} "
            f"{classification:<11} "
            f"{probability:>6.2f}%"
        )