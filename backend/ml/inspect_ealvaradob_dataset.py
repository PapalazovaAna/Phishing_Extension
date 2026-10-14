from pathlib import Path
import json
from collections import Counter


DATASET_PATH = (
    Path(__file__).resolve().parents[1]
    / "app"
    / "ml"
    / "data"
    / "ealvaradob"
    / "urls.json"
)


def main():
    print(f"Loading:\n{DATASET_PATH}")

    with DATASET_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    print(f"\nLoaded object type: {type(data)}")

    if isinstance(data, dict):
        print(f"Top-level keys: {list(data.keys())}")

        if "data" in data:
            records = data["data"]
        else:
            records = data

    elif isinstance(data, list):
        records = data

    else:
        raise ValueError(
            f"Unsupported JSON type: {type(data)}"
        )

    if isinstance(records, dict):
        records = list(records.values())

    print(f"Number of records: {len(records)}")

    if not records:
        raise ValueError("Dataset is empty")

    print("\nFirst record:")
    print(records[0])

    if isinstance(records[0], dict):
        print(
            "\nRecord keys:",
            list(records[0].keys()),
        )

    label_counter = Counter()
    text_lengths = []

    for record in records:
        if not isinstance(record, dict):
            continue

        label = record.get("label")
        text = record.get("text")

        label_counter[str(label)] += 1

        if text is not None:
            text_lengths.append(len(str(text)))

    print("\nLabel distribution:")

    for label, count in label_counter.items():
        print(f"  Label {label}: {count}")

    if text_lengths:
        print("\nText statistics:")
        print(f"  Minimum length: {min(text_lengths)}")
        print(f"  Maximum length: {max(text_lengths)}")
        print(
            f"  Average length: "
            f"{sum(text_lengths) / len(text_lengths):.2f}"
        )

    print("\nFirst 10 records:")

    for index, record in enumerate(records[:10]):
        print(f"\nRecord {index + 1}:")
        print(f"Text: {record.get('text')}")
        print(f"Label: {record.get('label')}")

    texts = [
        str(record.get("text"))
        for record in records
        if isinstance(record, dict)
    ]

    print("\nDuplicate statistics:")
    print(f"  Total records: {len(texts)}")
    print(f"  Unique texts: {len(set(texts))}")
    print(
        f"  Duplicate records: "
        f"{len(texts) - len(set(texts))}"
    )


if __name__ == "__main__":
    main()