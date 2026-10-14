from pathlib import Path

import json
import random

import numpy as np
import pandas as pd
import torch
import tldextract

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models" / "deberta-v3-small-url"

PHIUSIIL_PATH = DATA_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv"
LEGITPHISH_PATH = DATA_DIR / "url_features_extracted1.csv"

BASE_MODEL = "microsoft/deberta-v3-small"

MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 1
LEARNING_RATE = 2e-5
RANDOM_STATE = 42

# Остави None за цел dataset.
# Ако тренирањето е премногу бавно, стави, на пример, 100_000.
MAX_RECORDS = None


EXTERNAL_URLS = {
    "Google": (
        "https://www.google.com/",
        0,
    ),
    "Gmail": (
        "https://mail.google.com/mail/u/0/#inbox",
        0,
    ),
    "Gmail with query": (
        "https://mail.google.com/mail/u/0/?tab=rm&ogbl#inbox",
        0,
    ),
    "Booking": (
        "https://booking.com/payments",
        0,
    ),
    "Booking with query": (
        "https://www.booking.com/"
        "?chal_t=1790763960679&force_referer=",
        0,
    ),
    "FINKI": (
        "https://finki.ukim.mk/",
        0,
    ),
    "FINKI undergraduate studies": (
        "https://finki.ukim.mk/"
        "studii-2/vidovi-studii/dodiplomski-studii/",
        0,
    ),
    "FINKI masters quotas": (
        "https://finki.ukim.mk/"
        "upisi/magisterski-studii/kvoti/",
        0,
    ),
    "Microsoft": (
        "https://www.microsoft.com/",
        0,
    ),
    "GitHub": (
        "https://github.com/",
        0,
    ),
    "Python documentation": (
        "https://docs.python.org/3/library/urllib.parse.html",
        0,
    ),
    "Typosquatting Microsoft": (
        "https://rnicrosoft.com/",
        1,
    ),
    "Suspicious login": (
        "http://secure-login-verification.example/login",
        1,
    ),
    "Suspicious account": (
        "http://account-confirmation.example/verify/password",
        1,
    ),
    "Suspicious payment": (
        "http://paypal-secure.example/verify/account",
        1,
    ),
    "IP address URL": (
        "http://192.0.2.10/login",
        1,
    ),
    "Encoded phishing URL": (
        "http://example.invalid/%6cogin/%76erify/password",
        1,
    ),
}


def set_seed() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


def get_registered_domain(url: str) -> str:
    extracted = tldextract.extract(url)

    registered_domain = (
        extracted.top_domain_under_public_suffix
    )

    if registered_domain:
        return registered_domain.lower()

    return extracted.domain.lower()


def load_phiusiil() -> pd.DataFrame:
    dataframe = pd.read_csv(PHIUSIIL_PATH)

    dataframe = dataframe.rename(
        columns={
            "URL": "url",
            "label": "label",
        }
    )

    dataframe = dataframe[
        ["url", "label"]
    ].copy()

    dataframe["label"] = pd.to_numeric(
        dataframe["label"],
        errors="coerce",
    )

    # Raw PhiUSIIL:
    # 0 = phishing
    # 1 = legitimate
    #
    # Our common encoding:
    # 0 = legitimate
    # 1 = phishing
    dataframe["label"] = 1 - dataframe["label"]

    dataframe["source"] = "PhiUSIIL"

    return dataframe


def load_legitphish() -> pd.DataFrame:
    dataframe = pd.read_csv(LEGITPHISH_PATH)

    dataframe = dataframe.rename(
        columns={
            "URL": "url",
            "ClassLabel": "label",
        }
    )

    dataframe = dataframe[
        ["url", "label"]
    ].copy()

    dataframe["label"] = pd.to_numeric(
        dataframe["label"],
        errors="coerce",
    )

    # Raw LegitPhish:
    # 0 = phishing
    # 1 = legitimate
    dataframe["label"] = 1 - dataframe["label"]

    dataframe["source"] = "LegitPhish"

    return dataframe


def load_dataset() -> pd.DataFrame:
    phiusiil = load_phiusiil()
    legitphish = load_legitphish()

    dataframe = pd.concat(
        [phiusiil, legitphish],
        ignore_index=True,
    )

    dataframe["url"] = (
        dataframe["url"]
        .astype(str)
        .str.strip()
    )

    dataframe = dataframe.dropna(
        subset=["url", "label"]
    )

    dataframe["label"] = (
        dataframe["label"]
        .astype(int)
    )

    dataframe = dataframe[
        dataframe["label"].isin([0, 1])
    ]

    dataframe = dataframe[
        dataframe["url"].str.len() > 0
    ]

    dataframe = dataframe.drop_duplicates(
        subset=["url", "label"]
    )

    label_counts = (
        dataframe.groupby("url")["label"]
        .nunique()
    )

    conflicting_urls = label_counts[
        label_counts > 1
    ].index

    dataframe = dataframe[
        ~dataframe["url"].isin(conflicting_urls)
    ]

    dataframe["domain"] = dataframe["url"].apply(
        get_registered_domain
    )

    if MAX_RECORDS is not None:
        samples_per_class = MAX_RECORDS // 2

        sampled_groups = []

        for _, group in dataframe.groupby(
            "label",
            sort=True,
        ):
            sample_size = min(
                len(group),
                samples_per_class,
            )

            sampled_groups.append(
                group.sample(
                    n=sample_size,
                    random_state=RANDOM_STATE,
                )
            )

        dataframe = pd.concat(
            sampled_groups,
            ignore_index=True,
        )
    
    return dataframe.reset_index(drop=True)


class UrlDataset(Dataset):
    def __init__(
        self,
        urls: list[str],
        labels: list[int],
    ) -> None:
        self.urls = urls
        self.labels = labels

    def __len__(self) -> int:
        return len(self.urls)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[str, int]:
        return (
            self.urls[index],
            self.labels[index],
        )


def create_collate_fn(tokenizer):
    def collate_fn(batch):
        urls, labels = zip(*batch)

        tokenized = tokenizer(
            list(urls),
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        tokenized["labels"] = torch.tensor(
            labels,
            dtype=torch.long,
        )

        return tokenized

    return collate_fn


def evaluate_model(
    model,
    dataloader,
    device,
    class_weights,
):
    model.eval()

    all_labels = []
    all_predictions = []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            labels = batch.pop("labels").to(device)

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)

            loss = F.cross_entropy(
                outputs.logits.float(),
                labels,
                weight=class_weights.float(),
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

            predictions = torch.argmax(
                probabilities,
                dim=-1,
            )

            total_loss += loss.item()
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(
                predictions.cpu().tolist()
            )

    return (
        total_loss / max(len(dataloader), 1),
        all_labels,
        all_predictions,
    )


def evaluate_external_urls(
    model,
    tokenizer,
    device,
) -> None:
    print()
    print("=" * 70)
    print("EXTERNAL URL EVALUATION")
    print("=" * 70)

    legitimate_total = 0
    legitimate_false_positives = 0

    phishing_total = 0
    phishing_true_positives = 0

    model.eval()

    for name, (url, expected_label) in EXTERNAL_URLS.items():
        inputs = tokenizer(
            url,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1,
        )[0]

        phishing_probability = float(
            probabilities[1].item()
        )

        prediction = int(
            phishing_probability >= 0.50
        )

        prediction_name = (
            "phishing"
            if prediction == 1
            else "legitimate"
        )

        expected_name = (
            "phishing"
            if expected_label == 1
            else "legitimate"
        )

        print()
        print(name)
        print(f"URL: {url}")
        print(f"Expected: {expected_name}")
        print(f"Prediction: {prediction_name}")
        print(
            "Phishing probability: "
            f"{phishing_probability * 100:.2f}%"
        )

        if expected_label == 0:
            legitimate_total += 1

            if prediction == 1:
                legitimate_false_positives += 1
        else:
            phishing_total += 1

            if prediction == 1:
                phishing_true_positives += 1

    print()
    print("=" * 70)
    print("EXTERNAL SUMMARY")
    print("=" * 70)
    print(
        "Legitimate false positives: "
        f"{legitimate_false_positives}/"
        f"{legitimate_total}"
    )
    print(
        "Phishing true positives: "
        f"{phishing_true_positives}/"
        f"{phishing_total}"
    )


def main() -> None:
    set_seed()

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"Using device: {device}")

    dataframe = load_dataset()

    print()
    print("Dataset shape:")
    print(dataframe.shape)

    print()
    print("Source distribution:")
    print(dataframe["source"].value_counts())

    print()
    print("Normalized class distribution:")
    print(
        dataframe["label"]
        .map({
            0: "legitimate",
            1: "phishing",
        })
        .value_counts()
    )

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_STATE,
    )

    train_indices, test_indices = next(
        splitter.split(
            dataframe["url"],
            dataframe["label"],
            groups=dataframe["domain"],
        )
    )

    train_data = dataframe.iloc[
        train_indices
    ].copy()

    test_data = dataframe.iloc[
        test_indices
    ].copy()

    print()
    print(f"Training records: {len(train_data)}")
    print(f"Test records: {len(test_data)}")
    print(
        "Training domains: "
        f"{train_data['domain'].nunique()}"
    )
    print(
        "Test domains: "
        f"{test_data['domain'].nunique()}"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL
    )

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            BASE_MODEL,
            num_labels=2,
            id2label={
                0: "LEGITIMATE",
                1: "PHISHING",
            },
            label2id={
                "LEGITIMATE": 0,
                "PHISHING": 1,
            },
        )
    )

    model = model.to(device)
    model = model.float()

    train_dataset = UrlDataset(
        urls=train_data["url"].tolist(),
        labels=train_data["label"].tolist(),
    )

    test_dataset = UrlDataset(
        urls=test_data["url"].tolist(),
        labels=test_data["label"].tolist(),
    )

    collate_fn = create_collate_fn(tokenizer)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    class_counts = (
        train_data["label"]
        .value_counts()
        .sort_index()
    )

    class_weights = torch.tensor(
        [
            len(train_data)
            / (2 * class_counts[0]),
            len(train_data)
            / (2 * class_counts[1]),
        ],
        dtype=torch.float32,
        device=device,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    print()
    print("Training DeBERTa URL classifier...")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_number, batch in enumerate(
            train_loader,
            start=1,
        ):
            labels = batch.pop("labels").to(device)

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(**batch)

            loss = F.cross_entropy(
                outputs.logits.float(),
                labels,
                weight=class_weights.float(),
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_number % 500 == 0:
                print(
                    f"Epoch {epoch + 1}, "
                    f"batch {batch_number}, "
                    f"loss {loss.item():.4f}"
                )

        validation_loss, labels, predictions = (
            evaluate_model(
                model=model,
                dataloader=test_loader,
                device=device,
                class_weights=class_weights,
            )
        )

        print()
        print(f"Epoch {epoch + 1} completed")
        print(
            "Training loss: "
            f"{total_loss / len(train_loader):.4f}"
        )
        print(
            f"Validation loss: {validation_loss:.4f}"
        )
        print(
            f"Validation accuracy: "
            f"{accuracy_score(labels, predictions):.4f}"
        )

    print()
    print("=" * 70)
    print("FINAL CLASSIFICATION REPORT")
    print("=" * 70)

    print(
        classification_report(
            labels,
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
            labels,
            predictions,
        )
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    metadata = {
        "base_model": BASE_MODEL,
        "max_length": MAX_LENGTH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "label_mapping": {
            "0": "legitimate",
            "1": "phishing",
        },
    }

    with open(
        MODEL_DIR / "training_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print()
    print(f"Model saved to: {MODEL_DIR}")

    evaluate_external_urls(
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


if __name__ == "__main__":
    main()