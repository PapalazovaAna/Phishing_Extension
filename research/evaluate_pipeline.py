"""Evaluate the complete scoring pipeline instead of the ML model alone.

Reads the prediction files written by evaluate_urlbert_vonpower.py and
evaluate_urlbert_ealvaradob.py and computes, for every test URL, the
final score the backend would return: URL heuristics, ML probability,
reputation and typosquatting, combined by app/scoring.py.

The datasets contain only URLs, so the DOM score is 0. This is the
situation of the automatic check when a page is opened, which also
sends no DOM data.

Run from the repository root, after the evaluation scripts:

    python -m research.evaluate_pipeline
"""

import csv
import json
import re
from collections import Counter
from urllib.parse import urlparse

from app.reputation import (
    check_reputation,
    load_reputation_data,
)
from app.scoring import (
    classify_score,
    combine_scores,
    should_block,
)
from research import RESULTS_DIR


# Prediction files to evaluate: (model, dataset) -> file name.
PREDICTION_FILES = {
    ("vonpower_full", "vonpower"): (
        "urlbert_vonpower_full_vonpower_test_predictions.csv"
    ),
    ("vonpower_full", "ealvaradob"): (
        "urlbert_vonpower_full_ealvaradob_test_https_predictions.csv"
    ),
    ("combined_v2", "vonpower"): (
        "urlbert_combined_v2_vonpower_test_predictions.csv"
    ),
    ("combined_v2", "ealvaradob"): (
        "urlbert_combined_v2_ealvaradob_test_https_predictions.csv"
    ),
}

SUSPICIOUS_WORDS = ["login", "verify", "secure", "account", "password"]

IP_ADDRESS = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")


def analyze_url(url):
    """Python port of analyzeUrl in extension/service-worker.js."""

    try:
        parsed_url = urlparse(url)
        hostname = parsed_url.hostname or ""
    except ValueError:
        return 100

    score = 0

    if parsed_url.scheme != "https":
        score += 10

    if IP_ADDRESS.match(hostname):
        score += 25

    if len(url) > 100:
        score += 10

    if len(hostname.split(".")) > 3:
        score += 10

    if any(word in url.lower() for word in SUSPICIOUS_WORDS):
        score += 15

    return min(score, 100)


def evaluate_file(path):
    counts = {
        label: Counter()
        for label in (0, 1)
    }

    with path.open(encoding="utf-8") as file:
        for row in csv.DictReader(file):
            url = row["url"]
            label = int(row["true_label"])
            ml_score = round(float(row["phishing_probability"]) * 100)

            url_score = analyze_url(url)
            reputation = check_reputation(url)

            final_score = combine_scores(
                url_score=url_score,
                ml_score=ml_score,
                dom_score=0,
                reputation_status=reputation["status"],
            )

            blocked = should_block(
                final_score=final_score,
                url_score=url_score,
                ml_score=ml_score,
                reputation_status=reputation["status"],
            )

            for group in ("all", reputation["status"]):
                counts[label][(group, "total")] += 1
                counts[label][(group, classify_score(final_score))] += 1
                counts[label][(group, "blocked")] += int(blocked)

    results = {}

    for label, name in ((0, "legitimate"), (1, "phishing")):
        groups = sorted(
            {group for group, _ in counts[label]}
        )

        results[name] = {}

        for group in groups:
            total = counts[label][(group, "total")]

            results[name][group] = {
                "count": total,
                "blocked": round(
                    counts[label][(group, "blocked")] / total,
                    4,
                ),
                **{
                    classification: round(
                        counts[label][(group, classification)] / total,
                        4,
                    )
                    for classification in (
                        "Likely safe",
                        "Suspicious",
                        "Likely phishing",
                    )
                },
            }

    return results


def main():
    load_reputation_data()

    results = {}

    for (model, dataset), file_name in PREDICTION_FILES.items():
        path = RESULTS_DIR / file_name

        if not path.exists():
            print(f"Missing {path.name}, skipping.")
            continue

        print(f"\n{model} on {dataset}")
        print("=" * 70)

        results[f"{model}_{dataset}"] = evaluate_file(path)

        for name, groups in results[f"{model}_{dataset}"].items():
            for group, values in groups.items():
                print(
                    f"{name:10} {group:10} n={values['count']:6} "
                    f"blocked {values['blocked']:.2%}, "
                    f"suspicious {values['Suspicious']:.2%}, "
                    f"likely phishing {values['Likely phishing']:.2%}"
                )

    results_path = RESULTS_DIR / "pipeline_results.json"

    results_path.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
