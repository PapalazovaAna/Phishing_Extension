from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "app" / "ml" / "data"
OUTPUT_PATH = DATA_DIR / "external_evaluation_urls.csv"


DATASET_CANDIDATES = [
    DATA_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv",
    DATA_DIR / "url_features_extracted1.csv",
]


def find_column(columns: list[str], expected: str) -> str:
    for column in columns:
        if column.lower() == expected.lower():
            return column
    raise ValueError(
        f"Could not find '{expected}' in columns: {list(columns)}"
    )


def normalize_label(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")

    if numeric.notna().all():
        values = set(numeric.unique())

        if values.issubset({0, 1}):
            # Во dataset-ите:
            # 0 = phishing, 1 = legitimate
            #
            # Во проектот:
            # 0 = legitimate, 1 = phishing
            return (1 - numeric).astype(int)

    normalized = series.astype("string").str.lower().str.strip()

    mapping = {
        # Raw dataset: 0 = phishing
        "0": 1,
        "0.0": 1,

        # Raw dataset: 1 = legitimate
        "1": 0,
        "1.0": 0,

        "phishing": 1,
        "malicious": 1,
        "true": 1,

        "legitimate": 0,
        "benign": 0,
        "safe": 0,
        "false": 0,
    }

    result = normalized.map(mapping)

    unknown = normalized[
        result.isna() & normalized.notna()
    ].unique().tolist()

    if unknown:
        raise ValueError(f"Unknown labels: {unknown}")

    return result


def registered_domain(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower().strip(".")
    parts = hostname.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else hostname


def load_dataset(path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    url_column = find_column(dataframe.columns, "url")

    if "label" in {column.lower() for column in dataframe.columns}:
        label_column = find_column(dataframe.columns, "label")
    elif "classlabel" in {column.lower() for column in dataframe.columns}:
        label_column = find_column(dataframe.columns, "ClassLabel")
    else:
        raise ValueError(f"No label column found in {path}")

    result = dataframe[[url_column, label_column]].copy()
    result.columns = ["url", "label"]
    result["label"] = normalize_label(result["label"])
    result = result.dropna(subset=["url", "label"])
    result["label"] = result["label"].astype(int)
    result["source"] = path.stem
    return result


def main() -> None:
    existing = [path for path in DATASET_CANDIDATES if path.exists()]
    if not existing:
        raise FileNotFoundError(
            "No supported dataset was found. Expected one of: "
            + ", ".join(str(path) for path in DATASET_CANDIDATES)
        )

    datasets = [load_dataset(path) for path in existing]
    dataframe = pd.concat(datasets, ignore_index=True)
    dataframe["url"] = dataframe["url"].astype(str).str.strip()
    dataframe = dataframe[dataframe["url"] != ""]
    dataframe = dataframe.dropna(subset=["url", "label"])
    dataframe = dataframe.drop_duplicates(subset=["url"])
    dataframe["domain"] = dataframe["url"].map(registered_domain)

    splitter = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    _, external_indices = next(
        splitter.split(
            dataframe["url"],
            dataframe["label"],
            groups=dataframe["domain"],
        )
    )

    external = dataframe.iloc[external_indices].copy()
    external = external.drop(columns=["domain"])
    external = external.sort_values(["source", "label", "url"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    external.to_csv(OUTPUT_PATH, index=False)

    print("Full dataset records:", len(dataframe))
    print("External evaluation records:", len(external))
    print("External domains:", external["url"].map(registered_domain).nunique())
    print("Source distribution:")
    print(external["source"].value_counts())
    print("Class distribution:")
    print(external["label"].value_counts().sort_index())
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()