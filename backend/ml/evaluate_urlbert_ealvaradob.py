"""Evaluate the URLBERT model on a second, independent dataset.

The model was fine-tuned on the vonpower dataset (2018). This
script measures how well it generalises to the ealvaradob
phishing dataset, which was collected from different sources.

The dataset has no official split. The test split is selected by
a hash of the URL, the same way as in train_urlbert_combined.ipynb,
so models trained with that notebook never saw these URLs.

Most legitimate URLs in ealvaradob have no scheme, while many
phishing URLs use https://. To prevent the scheme from deciding
the result, every URL is rewritten to the same scheme (--scheme).
URLs that also appear in the vonpower training data are removed.

Run from the backend directory:

    python -m ml.download_ealvaradob_dataset
    python -m ml.evaluate_urlbert_ealvaradob --scheme https
    python -m ml.evaluate_urlbert_ealvaradob --scheme http
    python -m ml.evaluate_urlbert_ealvaradob --model-dir app/ml/models/urlbert_combined_v2
"""

from pathlib import Path
import argparse
import hashlib
import json

import dill
import numpy as np

from ml.evaluate_urlbert_vonpower import (
    DATASET_PATH as VONPOWER_PATH,
    RESULTS_DIR,
    compute_metrics,
    load_model,
    load_split,
    model_label,
    predict_probabilities,
    print_metrics,
    save_predictions,
    select_balanced_subset,
)


BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    BASE_DIR
    / "app"
    / "ml"
    / "data"
    / "ealvaradob"
    / "urls.json"
)


def ealvaradob_split(url):
    # Must match ealvaradob_split in train_urlbert_combined.ipynb.
    bucket = int(
        hashlib.md5(url.encode("utf-8")).hexdigest(),
        16,
    ) % 10

    if bucket == 0:
        return "test"

    if bucket == 1:
        return "val"

    return "train"


def strip_scheme(url):
    for scheme in ("https://", "http://"):
        if url.lower().startswith(scheme):
            return url[len(scheme):]

    return url


def load_ealvaradob():
    with DATASET_PATH.open("r", encoding="utf-8") as file:
        records = json.load(file)

    urls = [
        strip_scheme(record["text"].strip())
        for record in records
    ]

    labels = np.array(
        [int(record["label"]) for record in records]
    )

    return urls, labels


def load_vonpower_train_urls():
    with VONPOWER_PATH.open("rb") as file:
        data = dill.load(file)

    return {
        strip_scheme(url)
        for url in load_split(data, "train")[0]
    }


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
        "--scheme",
        choices=["https", "http"],
        default="https",
        help="Scheme added to every URL after removing the original.",
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

    args = parser.parse_args()

    print(f"Loading dataset from:\n{DATASET_PATH}")

    urls, labels = load_ealvaradob()
    original_count = len(urls)

    # Remove empty URLs, duplicates with conflicting labels and
    # URLs seen during fine-tuning.
    train_urls = load_vonpower_train_urls()

    url_labels = {}

    for url, label in zip(urls, labels):
        url_labels.setdefault(url, set()).add(int(label))

    kept = [
        (url, next(iter(url_labels[url])))
        for url in dict.fromkeys(urls)
        if url
        and len(url_labels[url]) == 1
        and ealvaradob_split(url) == args.split
        and url not in train_urls
    ]

    urls = [url for url, _ in kept]
    labels = np.array([label for _, label in kept])

    print(
        f"Records: {original_count}, {args.split} split without "
        f"duplicates, conflicting labels and vonpower training "
        f"URLs: {len(urls)}"
    )

    urls, labels = select_balanced_subset(
        urls,
        labels,
        args.max_per_class,
    )

    urls = [f"{args.scheme}://{url}" for url in urls]

    model, tokenizer, model_name, preprocess = load_model(
        args.model,
        args.model_dir,
    )

    print(f"\nModel: {model_name}")
    print(f"Scoring {len(urls)} URLs with {args.scheme}://...")

    probabilities = predict_probabilities(
        urls,
        model,
        tokenizer,
        args.batch_size,
        preprocess,
    )

    results = {
        "model": model_name,
        "dataset": "ealvaradob/phishing-dataset urls.json",
        "split": args.split,
        "scheme": args.scheme,
        "max_per_class": args.max_per_class,
        "records_after_filtering": len(kept),
        "all": compute_metrics(labels, probabilities),
    }

    print_metrics(
        f"ealvaradob, all URLs rewritten to {args.scheme}://",
        results["all"],
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    label = model_label(args.model, args.model_dir)

    file_stem = (
        f"urlbert_{label}_ealvaradob_{args.split}_{args.scheme}"
    )

    results_path = RESULTS_DIR / f"{file_stem}.json"

    results_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    save_predictions(
        RESULTS_DIR / f"{file_stem}_predictions.csv",
        urls,
        labels,
        probabilities,
        np.zeros(len(urls), dtype=bool),
    )

    print(f"\nMetrics saved to: {results_path}")


if __name__ == "__main__":
    main()
