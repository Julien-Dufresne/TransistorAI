
# Day 8 — Prepare reproducible E2 mixed-domain datasets.
# E2 combines technical and general German-English translation pairs.

import json
import hashlib
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# 1. Define project paths and reproducibility settings.
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

TECH_DIR = ROOT / "data/processed/technical/de-en"
GENERAL_DIR = ROOT / "data/processed/general/de-en"
OUTPUT_DIR = ROOT / "data/processed/mixed/de-en"

SEED = 42

# Match the verified E1 technical training row count.
TRAIN_BUDGET = 101_401


def load_dataset(path):
    """Load and validate a parallel translation dataset."""

    df = pd.read_csv(path, usecols=["source", "target"])

    # Remove missing and empty translations.
    df = df.dropna(subset=["source", "target"]).copy()

    for column in ["source", "target"]:
        df[column] = df[column].astype(str).str.strip()

    df = df[
        (df["source"] != "") &
        (df["target"] != "")
    ]

    # Remove exact duplicate translation pairs.
    df = df.drop_duplicates(subset=["source", "target"])

    return df.reset_index(drop=True)


def fingerprint(path):
    """Calculate a SHA-256 hash for dataset reproducibility."""

    hasher = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            hasher.update(chunk)

    return hasher.hexdigest()


def pairs(df):
    """Return normalized source-target translation pairs."""

    return set(zip(df["source"], df["target"]))


def main():
    # --------------------------------------------------------
    # 2. Load the technical and general datasets.
    # --------------------------------------------------------

    technical_train = load_dataset(TECH_DIR / "train.csv")
    general_train = load_dataset(GENERAL_DIR / "train.csv")

    technical_val = load_dataset(TECH_DIR / "validation.csv")
    general_val = load_dataset(GENERAL_DIR / "validation.csv")

    technical_test = load_dataset(TECH_DIR / "technical_test.csv")
    general_test = load_dataset(GENERAL_DIR / "general_test.csv")

    # --------------------------------------------------------
    # 3. Determine the E2 sampling budget.
    # --------------------------------------------------------

    # Use every available general training example.
    general_count = len(general_train)

    # Fill the remaining training budget with technical data.
    technical_count = TRAIN_BUDGET - general_count

    if technical_count <= 0:
        raise ValueError("General dataset exceeds training budget.")

    if technical_count > len(technical_train):
        raise ValueError("Not enough technical training examples.")

    # Sample technical examples without replacement.
    technical_sample = technical_train.sample(
        n=technical_count,
        random_state=SEED,
        replace=False,
    )

    # Combine technical and general training examples.
    mixed_train = pd.concat(
        [technical_sample, general_train],
        ignore_index=True,
    )

    # Shuffle the final dataset reproducibly.
    mixed_train = mixed_train.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 4. Combine the validation datasets.
    # --------------------------------------------------------

    mixed_validation = pd.concat(
        [technical_val, general_val],
        ignore_index=True,
    )

    mixed_validation = mixed_validation.sample(
        frac=1,
        random_state=SEED,
    ).reset_index(drop=True)

    # --------------------------------------------------------
    # 5. Verify that E2 contains no exact-pair leakage.
    # --------------------------------------------------------

    train_pairs = pairs(mixed_train)
    validation_pairs = pairs(mixed_validation)
    test_pairs = pairs(technical_test) | pairs(general_test)

    # Ensure there are no duplicate translation pairs.
    if len(train_pairs) != len(mixed_train):
        raise ValueError("Duplicate training pairs detected.")

    if len(validation_pairs) != len(mixed_validation):
        raise ValueError("Duplicate validation pairs detected.")

    # Prevent training on validation or held-out test pairs.
    if train_pairs & validation_pairs:
        raise ValueError("Train/validation overlap detected.")

    if train_pairs & test_pairs:
        raise ValueError("Train/test leakage detected.")

    if validation_pairs & test_pairs:
        raise ValueError("Validation/test leakage detected.")

    # --------------------------------------------------------
    # 6. Save the prepared E2 datasets.
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / "train.csv"
    validation_path = OUTPUT_DIR / "validation.csv"

    mixed_train.to_csv(train_path, index=False)
    mixed_validation.to_csv(validation_path, index=False)

    # --------------------------------------------------------
    # 7. Save a reproducibility manifest.
    # --------------------------------------------------------

    manifest = {
        "experiment": "E2 mixed-domain fine-tuning",
        "language_pair": "de-en",
        "seed": SEED,
        "training_budget": TRAIN_BUDGET,
        "technical_training_examples": technical_count,
        "general_training_examples": general_count,
        "total_training_examples": len(mixed_train),
        "technical_validation_examples": len(technical_val),
        "general_validation_examples": len(general_val),
        "total_validation_examples": len(mixed_validation),
        "sampling": "technical without replacement",
        "test_sets_used_for_training": False,
        "train_sha256": fingerprint(train_path),
        "validation_sha256": fingerprint(validation_path),
    }

    manifest_path = OUTPUT_DIR / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    # --------------------------------------------------------
    # 8. Display a preparation summary.
    # --------------------------------------------------------

    print("\nE2 dataset preparation completed.")
    print(f"Technical training: {technical_count:,}")
    print(f"General training: {general_count:,}")
    print(f"Total training: {len(mixed_train):,}")
    print(f"Total validation: {len(mixed_validation):,}")
    print("Exact-pair leakage checks: PASSED")
    print(f"Manifest saved: {manifest_path}")


if __name__ == "__main__":
    main()
