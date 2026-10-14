"""Evaluate the URLBERT model on the held-out vonpower test split.

By default this evaluates the deployed model through the same
chunking and logit aggregation used by app/ml/predictor.py.
Pass --model base to evaluate the original pre-trained
CrabInHoney model for comparison, or --model-dir to evaluate
any other local model (for example a newly trained one).

Run from the backend directory:

    python -m ml.evaluate_urlbert_vonpower
    python -m ml.evaluate_urlbert_vonpower --model base
    python -m ml.evaluate_urlbert_vonpower --model-dir app/ml/models/urlbert_combined_v2
"""

from pathlib import Path
import argparse
import csv
import json
import time

import dill
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from app.ml import predictor


BASE_MODEL_NAME = (
    "CrabInHoney/"
    "urlbert-tiny-v4-phishing-classifier"
)

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR
    / "app"
    / "ml"
    / "data"
    / "vonDataset20180426.dill"
)

RESULTS_DIR = BASE_DIR / "ml" / "results"

# 0.50 is the standard decision threshold, 0.70 is the
# "Likely phishing" threshold used by the predictor.
THRESHOLDS = [0.50, 0.70]

# URLs in the dataset are stored with a fixed width of 150
# characters, so URLs of this length are probably truncated.
DATASET_URL_WIDTH = 150

RANDOM_SEED = 42


def decode_urls(encoded_urls, char_to_int):
    encoded_urls = np.asarray(encoded_urls)

    lookup = np.full(
        max(int(value) for value in char_to_int.values()) + 1,
        "",
        dtype="<U1",
    )

    for character, value in char_to_int.items():
        lookup[int(value)] = str(character)

    # Token 0 is padding.
    lookup[0] = ""

    characters = lookup[encoded_urls]

    rows = characters.view(
        f"<U{encoded_urls.shape[1]}"
    ).reshape(-1)

    return [
        row.replace("\x00", "")
        for row in rows
    ]


def load_split(data, split):
    urls = decode_urls(
        data[f"{split}_x"],
        data["char_to_int"],
    )

    labels = np.asarray(
        data[f"{split}_y"]
    ).reshape(-1)

    return urls, labels


def select_balanced_subset(urls, labels, max_per_class):
    if not max_per_class:
        return urls, labels

    rng = np.random.default_rng(RANDOM_SEED)

    selected_indices = []

    for label in [0, 1]:
        label_indices = np.flatnonzero(labels == label)

        selected_indices.extend(
            rng.choice(
                label_indices,
                size=min(max_per_class, len(label_indices)),
                replace=False,
            ).tolist()
        )

    selected_indices.sort()

    return (
        [urls[index] for index in selected_indices],
        labels[selected_indices],
    )


def no_preprocessing(url):
    return url


def load_model(model_choice, model_dir=None):
    """Returns the model, tokenizer, a name for the results and
    the URL preprocessing function of the model."""

    if model_dir:
        model_dir = Path(model_dir).resolve()
        preprocessing = predictor.load_preprocessing(model_dir)

        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir),
            local_files_only=True,
        )

        model.eval()

        return (
            model,
            tokenizer,
            str(model_dir),
            lambda url: predictor.preprocess_url(url, preprocessing),
        )

    if model_choice == "deployed":
        return (
            predictor.model,
            predictor.tokenizer,
            str(predictor.MODEL_DIRECTORY),
            predictor.preprocess_url,
        )

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_NAME
    )

    model.eval()

    return model, tokenizer, BASE_MODEL_NAME, no_preprocessing


def model_label(model_choice, model_dir=None):
    # Used in the names of the result files.
    if model_dir:
        return Path(model_dir).resolve().name.removeprefix("urlbert_")

    return model_choice


def predict_probabilities(
    urls,
    model,
    tokenizer,
    batch_size,
    preprocess=no_preprocessing,
):
    max_length = int(
        model.config.max_position_embeddings
    )

    probabilities = np.zeros(len(urls))
    start_time = time.time()

    for start in range(0, len(urls), batch_size):
        batch_urls = urls[start:start + batch_size]

        chunked = [
            predictor.create_url_chunks(
                preprocess(url),
                url_tokenizer=tokenizer,
                max_length=max_length,
            )
            for url in batch_urls
        ]

        chunk_counts = [
            len(input_ids)
            for input_ids, _ in chunked
        ]

        with torch.no_grad():
            logits = model(
                input_ids=torch.cat(
                    [input_ids for input_ids, _ in chunked]
                ),
                attention_mask=torch.cat(
                    [mask for _, mask in chunked]
                ),
            ).logits

        # Mean logits across the chunks of each URL,
        # the same aggregation as predictor.predict_probability.
        url_logits = torch.stack(
            [
                chunk_logits.mean(dim=0)
                for chunk_logits in torch.split(
                    logits,
                    chunk_counts,
                )
            ]
        )

        probabilities[start:start + len(batch_urls)] = (
            torch.softmax(url_logits, dim=1)[:, 1].numpy()
        )

        processed = start + len(batch_urls)

        if (
            processed % (batch_size * 40) == 0
            or processed == len(urls)
        ):
            elapsed = time.time() - start_time

            print(
                f"  {processed}/{len(urls)} URLs "
                f"({processed / elapsed:.0f} URLs/s)"
            )

    return probabilities


def compute_metrics(labels, probabilities):
    metrics = {
        "count": int(len(labels)),
        "legitimate": int(np.sum(labels == 0)),
        "phishing": int(np.sum(labels == 1)),
        "roc_auc": round(
            float(roc_auc_score(labels, probabilities)),
            4,
        ),
        "thresholds": {},
    }

    for threshold in THRESHOLDS:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            labels,
            predictions,
            labels=[0, 1],
        ).ravel()

        metrics["thresholds"][f"{threshold:.2f}"] = {
            "accuracy": round(
                float(accuracy_score(labels, predictions)),
                4,
            ),
            "precision": round(
                float(
                    precision_score(
                        labels,
                        predictions,
                        zero_division=0,
                    )
                ),
                4,
            ),
            "recall": round(
                float(recall_score(labels, predictions)),
                4,
            ),
            "f1": round(
                float(f1_score(labels, predictions)),
                4,
            ),
            "false_positive_rate": round(
                float(fp / (fp + tn)),
                4,
            ),
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            },
        }

    tiers = np.array(
        [
            predictor.classify_probability(probability)
            for probability in probabilities
        ]
    )

    metrics["predictor_tiers"] = {
        name: {
            tier: int(np.sum(tiers[labels == label] == tier))
            for tier in [
                "Likely safe",
                "Suspicious",
                "Likely phishing",
            ]
        }
        for label, name in [
            (0, "legitimate"),
            (1, "phishing"),
        ]
    }

    return metrics


def swap_scheme(url, old_scheme, new_scheme):
    return new_scheme + url[len(old_scheme):]


def evaluate_scheme_robustness(
    urls,
    labels,
    probabilities,
    model,
    tokenizer,
    batch_size,
    preprocess=no_preprocessing,
):
    """Re-score URLs with the scheme swapped.

    In this 2018 dataset almost all phishing URLs use http://,
    while modern phishing pages mostly use https://. This shows
    how much the model relies on the scheme alone.
    """

    results = {}

    cases = [
        (
            "phishing_http_to_https",
            1,
            "http://",
            "https://",
        ),
        (
            "legitimate_https_to_http",
            0,
            "https://",
            "http://",
        ),
    ]

    for name, label, old_scheme, new_scheme in cases:
        indices = [
            index
            for index, url in enumerate(urls)
            if labels[index] == label
            and url.startswith(old_scheme)
        ]

        swapped_urls = [
            swap_scheme(urls[index], old_scheme, new_scheme)
            for index in indices
        ]

        print(
            f"\nScoring {len(swapped_urls)} URLs "
            f"for {name}..."
        )

        swapped_probabilities = predict_probabilities(
            swapped_urls,
            model,
            tokenizer,
            batch_size,
            preprocess,
        )

        original_probabilities = probabilities[indices]

        results[name] = {
            "count": len(indices),
            "mean_probability_original": round(
                float(original_probabilities.mean()),
                4,
            ),
            "mean_probability_swapped": round(
                float(swapped_probabilities.mean()),
                4,
            ),
            "flagged_at_0.50_original": round(
                float(np.mean(original_probabilities >= 0.50)),
                4,
            ),
            "flagged_at_0.50_swapped": round(
                float(np.mean(swapped_probabilities >= 0.50)),
                4,
            ),
        }

    return results


def print_metrics(title, metrics):
    print(f"\n{title}")
    print("=" * 70)
    print(
        f"URLs: {metrics['count']} "
        f"(legitimate {metrics['legitimate']}, "
        f"phishing {metrics['phishing']})"
    )
    print(f"ROC-AUC: {metrics['roc_auc']}")

    for threshold, values in metrics["thresholds"].items():
        print(
            f"Threshold {threshold}: "
            f"accuracy {values['accuracy']}, "
            f"precision {values['precision']}, "
            f"recall {values['recall']}, "
            f"F1 {values['f1']}, "
            f"FPR {values['false_positive_rate']}"
        )
        print(f"  Confusion matrix: {values['confusion_matrix']}")

    print(f"Predictor tiers: {metrics['predictor_tiers']}")


def save_predictions(path, urls, labels, probabilities, seen):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "url",
                "true_label",
                "phishing_probability",
                "seen_in_train",
            ]
        )

        for row in zip(urls, labels, probabilities, seen):
            url, label, probability, was_seen = row

            writer.writerow(
                [
                    url,
                    int(label),
                    round(float(probability), 6),
                    int(was_seen),
                ]
            )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=["deployed", "base"],
        default="deployed",
    )
    parser.add_argument(
        "--model-dir",
        help="Evaluate a model from this directory instead.",
    )
    parser.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=0,
        help="Balanced subsample size per class, 0 = whole split.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--skip-scheme-check",
        action="store_true",
    )

    args = parser.parse_args()

    print(f"Loading dataset from:\n{DATASET_PATH}")

    with DATASET_PATH.open("rb") as file:
        data = dill.load(file)

    urls, labels = load_split(data, args.split)

    urls, labels = select_balanced_subset(
        urls,
        labels,
        args.max_per_class,
    )

    train_urls = set(load_split(data, "train")[0])

    del data

    seen = np.array(
        [url in train_urls for url in urls]
    )

    truncated = np.array(
        [len(url) >= DATASET_URL_WIDTH for url in urls]
    )

    model, tokenizer, model_name, preprocess = load_model(
        args.model,
        args.model_dir,
    )

    print(f"\nModel: {model_name}")
    print(f"Scoring {len(urls)} {args.split} URLs...")

    probabilities = predict_probabilities(
        urls,
        model,
        tokenizer,
        args.batch_size,
        preprocess,
    )

    results = {
        "model": model_name,
        "dataset": DATASET_PATH.name,
        "split": args.split,
        "max_per_class": args.max_per_class,
        "urls_seen_in_train": int(seen.sum()),
        "urls_probably_truncated": int(truncated.sum()),
        "all": compute_metrics(labels, probabilities),
        "unseen_in_train": compute_metrics(
            labels[~seen],
            probabilities[~seen],
        ),
    }

    print_metrics(
        f"{args.split} split, all URLs",
        results["all"],
    )
    print_metrics(
        f"{args.split} split, URLs not present in train",
        results["unseen_in_train"],
    )

    if not args.skip_scheme_check:
        results["scheme_robustness"] = (
            evaluate_scheme_robustness(
                urls,
                labels,
                probabilities,
                model,
                tokenizer,
                args.batch_size,
                preprocess,
            )
        )

        print("\nScheme robustness")
        print("=" * 70)

        for name, values in results["scheme_robustness"].items():
            print(f"{name}: {values}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    label = model_label(args.model, args.model_dir)

    file_stem = f"urlbert_{label}_vonpower_{args.split}"

    results_path = RESULTS_DIR / f"{file_stem}.json"

    results_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    predictions_path = (
        RESULTS_DIR / f"{file_stem}_predictions.csv"
    )

    save_predictions(
        predictions_path,
        urls,
        labels,
        probabilities,
        seen,
    )

    print(f"\nMetrics saved to: {results_path}")
    print(f"Predictions saved to: {predictions_path}")


if __name__ == "__main__":
    main()
