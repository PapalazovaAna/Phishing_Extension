from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

import math
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from app.ml.features import extract_url_features, feature_names


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "app" / "ml" / "data"
DATASET_CANDIDATES = [
    DATA_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv",
    DATA_DIR / "url_features_extracted1.csv",
]
EXTERNAL_PATH = DATA_DIR / "external_evaluation_urls.csv"
RESULTS_PATH = BASE_DIR / "ml" / "benchmark_results.csv"


def registered_domain(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower().strip(".")
    parts = hostname.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else hostname


def normalize_labels(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        # Raw PhiUSIIL convention: 0=phishing, 1=legitimate.
        # Project convention: 0=legitimate, 1=phishing.
        return (1 - numeric).astype(int)

    normalized = series.astype(str).str.lower().str.strip()
    return normalized.isin({"phishing", "malicious", "true", "1"}).astype(int)


def load_dataset(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    url_column = next(
        column for column in dataframe.columns
        if column.lower() == "url"
    )
    label_candidates = {column.lower(): column for column in dataframe.columns}
    label_column = label_candidates.get("label") or label_candidates.get("classlabel")
    if label_column is None:
        raise ValueError(f"No label column found in {path}")

    dataframe = dataframe[[url_column, label_column]].copy()
    dataframe.columns = ["url", "label"]
    dataframe["url"] = dataframe["url"].astype(str).str.strip()
    dataframe["label"] = normalize_labels(dataframe["label"])
    dataframe = dataframe[dataframe["url"] != ""]
    dataframe = dataframe.drop_duplicates(subset=["url"])
    dataframe["domain"] = dataframe["url"].map(registered_domain)
    dataframe["source"] = path.stem
    return dataframe.reset_index(drop=True)


def load_main_dataset() -> pd.DataFrame:
    existing = [path for path in DATASET_CANDIDATES if path.exists()]
    if not existing:
        raise FileNotFoundError(
            "No supported dataset was found. Expected one of: "
            + ", ".join(str(path) for path in DATASET_CANDIDATES)
        )

    return pd.concat(
        [load_dataset(path) for path in existing],
        ignore_index=True,
    ).drop_duplicates(subset=["url"]).reset_index(drop=True)


def scores_for(model, features):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]

    decision = model.decision_function(features)
    return pd.Series(decision).map(
        lambda value: 1.0 / (1.0 + math.exp(-max(min(value, 50), -50)))
    ).to_numpy()


def calculate_metrics(labels, predictions, scores) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_phishing": precision_score(labels, predictions, zero_division=0),
        "recall_phishing": recall_score(labels, predictions, zero_division=0),
        "f1_phishing": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores),
        "legitimate_recall": recall_score(
            labels, predictions, pos_label=0, zero_division=0
        ),
        "false_positive_rate": (
            confusion_matrix(labels, predictions, labels=[0, 1])[0, 1]
            / max(sum(labels == 0), 1)
        ),
    }


def build_tabular_models():
    scaled = lambda estimator: Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])

    return {
        "Logistic Regression": scaled(
            LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "Linear SVM": scaled(
            LinearSVC(class_weight="balanced", random_state=42)
        ),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=7, n_jobs=-1)),
        "Decision Tree": DecisionTreeClassifier(
            random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=250, random_state=42
        ),
        "MLP": scaled(
            MLPClassifier(
                hidden_layer_sizes=(64, 32),
                max_iter=80,
                early_stopping=True,
                random_state=42,
            )
        ),
    }


def evaluate_model(name, model, x_train, y_train, x_test, y_test, kind):
    started = perf_counter()
    model.fit(x_train, y_train)
    train_seconds = perf_counter() - started

    started = perf_counter()
    predictions = model.predict(x_test)
    scores = scores_for(model, x_test)
    inference_ms = (perf_counter() - started) * 1000 / len(y_test)

    metrics = calculate_metrics(y_test, predictions, scores)
    metrics.update({
        "model": name,
        "feature_type": kind,
        "train_seconds": train_seconds,
        "inference_ms_per_url": inference_ms,
    })

    print(f"\n{name} ({kind})")
    print(classification_report(
        y_test,
        predictions,
        target_names=["legitimate", "phishing"],
        digits=4,
    ))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, predictions))
    return metrics


def evaluate_external(model, features, labels) -> dict[str, float]:
    predictions = model.predict(features)
    scores = scores_for(model, features)
    metrics = calculate_metrics(labels, predictions, scores)
    return {f"external_{key}": value for key, value in metrics.items()}


def main() -> None:
    dataframe = load_main_dataset()

    external = None
    if EXTERNAL_PATH.exists():
        external = pd.read_csv(EXTERNAL_PATH)
        external["url"] = external["url"].astype(str).str.strip()
        external["label"] = external["label"].astype(int)
        external_urls = set(external["url"])
        dataframe = dataframe[~dataframe["url"].isin(external_urls)].copy()
        dataframe = dataframe.reset_index(drop=True)
        print("External holdout records:", len(external))
        print("Records excluded from training/evaluation split:", len(external_urls))
    features = pd.DataFrame(
        dataframe["url"].map(extract_url_features).tolist(),
        columns=feature_names(),
    )
    labels = dataframe["label"]
    groups = dataframe["domain"]

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train_indices, test_indices = next(
        splitter.split(features, labels, groups=groups)
    )

    x_train = features.iloc[train_indices]
    x_test = features.iloc[test_indices]
    y_train = labels.iloc[train_indices]
    y_test = labels.iloc[test_indices]

    print("Dataset shape:", dataframe.shape)
    print("Training records:", len(x_train))
    print("Test records:", len(x_test))
    print("Training domains:", groups.iloc[train_indices].nunique())
    print("Test domains:", groups.iloc[test_indices].nunique())

    results = []
    external_tabular_features = None
    if external is not None:
        external_tabular_features = pd.DataFrame(
            external["url"].map(extract_url_features).tolist(),
            columns=feature_names(),
        )

    for name, model in build_tabular_models().items():
        metrics = evaluate_model(
            name, model, x_train, y_train, x_test, y_test, "tabular"
        )
        if external is not None:
            metrics.update(evaluate_external(
                model,
                external_tabular_features,
                external["label"],
            ))
        results.append(metrics)

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=150_000,
        sublinear_tf=True,
    )
    x_train_url = vectorizer.fit_transform(dataframe.iloc[train_indices]["url"])
    x_test_url = vectorizer.transform(dataframe.iloc[test_indices]["url"])
    external_url_features = None
    if external is not None:
        external_url_features = vectorizer.transform(external["url"])

    tfidf_models = {
        "Character TF-IDF + Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Character TF-IDF + Linear SVM": LinearSVC(
            class_weight="balanced", random_state=42
        ),
    }

    for name, model in tfidf_models.items():
        metrics = evaluate_model(
            name, model, x_train_url, y_train, x_test_url, y_test, "char_tfidf"
        )
        if external is not None:
            metrics.update(evaluate_external(
                model,
                external_url_features,
                external["label"],
            ))
        results.append(metrics)

    result_frame = pd.DataFrame(results).sort_values(
        ["false_positive_rate", "f1_phishing"],
        ascending=[True, False],
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    result_frame.to_csv(RESULTS_PATH, index=False)

    print("\nFINAL MODEL COMPARISON")
    print(result_frame.to_string(index=False))
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
