from research import DATA_DIR

import dill
import numpy as np


DATASET_PATH = DATA_DIR / "vonDataset20180426.dill"


def describe(name, value):
    print(f"\n{name}")
    print(f"Type: {type(value)}")

    if hasattr(value, "shape"):
        print(f"Shape: {value.shape}")

    if hasattr(value, "dtype"):
        print(f"Dtype: {value.dtype}")


def labels_to_1d(labels):
    labels = np.asarray(labels)

    if labels.ndim == 2 and labels.shape[1] > 1:
        return np.argmax(labels, axis=1)

    return labels.reshape(-1)


def decode_sample(sample, char_to_int):
    reverse_mapping = {
        int(value): str(key)
        for key, value in char_to_int.items()
    }

    sample = np.asarray(sample)

    # Ako e one-hot encoded
    if sample.ndim == 2:
        sample = np.argmax(sample, axis=1)

    characters = []

    for token in sample:
        token = int(token)

        # Najčesto 0 e padding
        if token == 0:
            continue

        characters.append(reverse_mapping.get(token, "?"))

    return "".join(characters)


def print_label_distribution(name, labels):
    labels = labels_to_1d(labels)
    unique, counts = np.unique(labels, return_counts=True)

    print(f"\n{name} label distribution:")

    for label, count in zip(unique, counts):
        print(f"  Label {label}: {count}")


def main():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    print(f"Loading dataset from:\n{DATASET_PATH}")

    with DATASET_PATH.open("rb") as file:
        data = dill.load(file)

    print("\nDataset keys:")
    print(list(data.keys()))

    for key, value in data.items():
        describe(key, value)

    train_x = data["train_x"]
    train_y = data["train_y"]
    val_x = data["val_x"]
    val_y = data["val_y"]
    test_x = data["test_x"]
    test_y = data["test_y"]
    char_to_int = data["char_to_int"]

    print_label_distribution("Train", train_y)
    print_label_distribution("Validation", val_y)
    print_label_distribution("Test", test_y)

    print("\nCharacter vocabulary:")
    print(f"Vocabulary size: {len(char_to_int)}")
    print(f"First characters: {list(char_to_int.items())[:20]}")

    print("\nDecoded training samples:")

    train_labels = labels_to_1d(train_y)

    for index in range(min(5, len(train_x))):
        decoded_url = decode_sample(
            train_x[index],
            char_to_int,
        )

        print(f"\nSample {index + 1}")
        print(f"Label: {train_labels[index]}")
        print(f"URL: {decoded_url}")

    print("\nDecoded test samples:")

    test_labels = labels_to_1d(test_y)

    for index in range(min(5, len(test_x))):
        decoded_url = decode_sample(
            test_x[index],
            char_to_int,
        )

        print(f"\nSample {index + 1}")
        print(f"Label: {test_labels[index]}")
        print(f"URL: {decoded_url}")


if __name__ == "__main__":
    main()