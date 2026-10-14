from pathlib import Path
from time import perf_counter
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "app" / "ml" / "data"
EXTERNAL_PATH = DATA_DIR / "external_evaluation_urls.csv"
RESULTS_PATH = BASE_DIR / "ml" / "deep_benchmark_results.csv"
CNN_MODEL_PATH = BASE_DIR / "ml" / "cnn_url_model.pt"

SEED = 42
MAX_LENGTH = 200
BATCH_SIZE = 256
EPOCHS = 2
MAX_TRAIN_PER_CLASS = 50_000
DEVICE = torch.device("cpu")


def registered_domain(url: str) -> str:
    hostname = (urlparse(url).hostname or "").lower().strip(".")
    parts = hostname.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else hostname


def load_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    url_col = next(c for c in frame.columns if c.lower() == "url")
    labels = {c.lower(): c for c in frame.columns}
    label_col = labels.get("label") or labels.get("classlabel")
    if label_col is None:
        raise ValueError(f"No label column found in {path}")

    result = frame[[url_col, label_col]].copy()
    result.columns = ["url", "label"]
    result = result.dropna(subset=["url", "label"])
    result["url"] = result["url"].astype(str).str.strip()
    result = result[result["url"] != ""]

    numeric = pd.to_numeric(result["label"], errors="coerce")
    if numeric.notna().all() and set(numeric.unique()).issubset({0, 1}):
        result["label"] = (1 - numeric).astype(int)
    else:
        normalized = result["label"].astype(str).str.lower().str.strip()
        result["label"] = normalized.isin(
            {"phishing", "malicious", "true", "1"}
        ).astype(int)

    return result.drop_duplicates("url").reset_index(drop=True)


class URLDataset(Dataset):
    def __init__(self, urls, labels, vocabulary):
        self.urls = urls
        self.labels = np.asarray(labels, dtype=np.float32)
        self.vocabulary = vocabulary

    def __len__(self):
        return len(self.urls)

    def __getitem__(self, index):
        encoded = [
            self.vocabulary.get(character, 1)
            for character in self.urls[index][:MAX_LENGTH]
        ]
        encoded += [0] * (MAX_LENGTH - len(encoded))
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(
            self.labels[index], dtype=torch.float32
        )


class CNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.network = nn.Sequential(
            nn.Conv1d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(self.embedding(x).transpose(1, 2)).squeeze(1)


class RecurrentURLModel(nn.Module):
    def __init__(self, vocab_size, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(vocab_size, 32, padding_idx=0)
        self.rnn = nn.LSTM(
            32, 32, batch_first=True, bidirectional=bidirectional
        )
        size = 64 if bidirectional else 32
        self.output = nn.Linear(size, 1)

    def forward(self, x):
        _, (hidden, _) = self.rnn(self.embedding(x))
        if self.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        return self.output(hidden).squeeze(1)


def metrics(model, loader):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for x, y in loader:
            scores = torch.sigmoid(model(x))
            predictions.extend((scores >= 0.5).int().numpy())
            labels.extend(y.int().numpy())
    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision_phishing": precision_score(labels, predictions, zero_division=0),
        "recall_phishing": recall_score(labels, predictions, zero_division=0),
        "f1_phishing": f1_score(labels, predictions, zero_division=0),
        "legitimate_recall": recall_score(
            labels, predictions, pos_label=0, zero_division=0
        ),
    }


def train_model(model, train_loader):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.BCEWithLogitsLoss()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"  Epoch {epoch + 1}/{EPOCHS}, loss: {total_loss / len(train_loader):.4f}")


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    paths = [
        DATA_DIR / "PhiUSIIL_Phishing_URL_Dataset.csv",
        DATA_DIR / "url_features_extracted1.csv",
    ]
    frame = pd.concat([load_dataset(path) for path in paths if path.exists()])
    frame = frame.drop_duplicates("url").reset_index(drop=True)

    if EXTERNAL_PATH.exists():
        external = pd.read_csv(EXTERNAL_PATH)
        external["label"] = external["label"].astype(int)
        frame = frame[~frame["url"].isin(set(external["url"]))].copy()
    else:
        external = pd.DataFrame(columns=["url", "label"])

    frame["domain"] = frame["url"].map(registered_domain)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, test_idx = next(
        splitter.split(frame["url"], frame["label"], groups=frame["domain"])
    )
    train = frame.iloc[train_idx]
    test = frame.iloc[test_idx]

    # Keep the CPU experiment manageable while preserving class balance.
    train = pd.concat(
        [
            group.sample(
                min(len(group), MAX_TRAIN_PER_CLASS), random_state=SEED
            )
            for _, group in train.groupby("label")
        ],
        ignore_index=True,
    )

    characters = sorted(set("".join(train["url"].tolist())))
    vocabulary = {character: index + 2 for index, character in enumerate(characters)}
    vocab_size = len(vocabulary) + 2

    train_loader = DataLoader(
        URLDataset(train["url"].tolist(), train["label"], vocabulary),
        batch_size=BATCH_SIZE, shuffle=True,
    )
    test_loader = DataLoader(
        URLDataset(test["url"].tolist(), test["label"], vocabulary),
        batch_size=BATCH_SIZE,
    )
    external_loader = DataLoader(
        URLDataset(external["url"].tolist(), external["label"], vocabulary),
        batch_size=BATCH_SIZE,
    ) if len(external) else None

    models = {
        "1D CNN": CNN(vocab_size),
        "LSTM": RecurrentURLModel(vocab_size),
        "BiLSTM": RecurrentURLModel(vocab_size, bidirectional=True),
    }
    results = []
    print(f"Train records: {len(train)}")
    print(f"Test records: {len(test)}")
    print(f"External records: {len(external)}")

    for name, model in models.items():
        print(f"\n{name}")
        started = perf_counter()
        train_model(model, train_loader)
        row = metrics(model, test_loader)
        row["model"] = name
        row["train_seconds"] = perf_counter() - started
        if external_loader:
            row.update({f"external_{k}": v for k, v in metrics(model, external_loader).items()})
        results.append(row)
        print(row)

        if name == "1D CNN":
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "vocabulary": vocabulary,
                    "max_length": MAX_LENGTH,
                },
                CNN_MODEL_PATH,
            )
            print(f"CNN model saved to: {CNN_MODEL_PATH}")

    pd.DataFrame(results).to_csv(RESULTS_PATH, index=False)
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
