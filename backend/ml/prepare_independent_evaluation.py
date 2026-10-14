from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "app" / "ml" / "data"
OUTPUT_PATH = DATA_DIR / "independent_evaluation_urls.csv"

TRANCO_PATH = DATA_DIR / "tranco.csv"
PHISHTANK_PATH = DATA_DIR / "online-valid.csv"


def registered_domain(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower().strip(".")
    parts = hostname.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else hostname


def load_used_domains() -> set[str]:
    used = set()
    for path in [
        DATA_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv",
        DATA_DIR / "url_features_extracted1.csv",
    ]:
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        url_column = next(c for c in frame.columns if c.lower() == "url")
        used.update(frame[url_column].dropna().astype(str).map(registered_domain))
    return used


def load_tranco() -> pd.DataFrame:
    # Tranco CSV is normally rank,domain without a header.
    frame = pd.read_csv(TRANCO_PATH, header=None, names=["rank", "url"])
    frame["url"] = "https://" + frame["url"].astype(str).str.strip()
    frame["label"] = 0
    return frame[["url", "label"]]


def load_phishtank() -> pd.DataFrame:
    frame = pd.read_csv(PHISHTANK_PATH)
    url_column = next(c for c in frame.columns if c.lower() == "url")
    result = frame[[url_column]].copy()
    result.columns = ["url"]
    result["label"] = 1
    return result


def main() -> None:
    if not TRANCO_PATH.exists() or not PHISHTANK_PATH.exists():
        raise FileNotFoundError(
            "Place tranco.csv and online-valid.csv in backend/app/ml/data"
        )

    legitimate = load_tranco()
    phishing = load_phishtank()
    frame = pd.concat([legitimate, phishing], ignore_index=True)
    frame["url"] = frame["url"].astype(str).str.strip()
    frame = frame.drop_duplicates("url")
    frame["domain"] = frame["url"].map(registered_domain)

    used_domains = load_used_domains()
    frame = frame[~frame["domain"].isin(used_domains)].copy()

    # Keep a balanced, reproducible sample.
    legitimate = frame[frame["label"] == 0].sample(
        min(5000, (frame["label"] == 0).sum()), random_state=42
    )
    phishing = frame[frame["label"] == 1].sample(
        min(5000, (frame["label"] == 1).sum()), random_state=42
    )
    result = pd.concat([legitimate, phishing], ignore_index=True)
    result = result[["url", "label"]].sample(frac=1, random_state=42)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Total records: {len(result)}")
    print("Class distribution:")
    print(result["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()