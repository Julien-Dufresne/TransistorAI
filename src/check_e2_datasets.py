
from pathlib import Path
import pandas as pd

# Project root, independent of where the script is launched.
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/processed"

# Define all datasets used by E2.
paths = {
    "technical_train": DATA / "technical/de-en/train.csv",
    "technical_validation": DATA / "technical/de-en/validation.csv",
    "technical_test": DATA / "technical/de-en/technical_test.csv",
    "general_train": DATA / "general/de-en/train.csv",
    "general_validation": DATA / "general/de-en/validation.csv",
    "general_test": DATA / "general/de-en/general_test.csv",
}

def load_pairs(path):
    """Load and normalize translation pairs for overlap checks."""

    df = pd.read_csv(path, usecols=["source", "target"])

    # Remove missing translations.
    df = df.dropna(subset=["source", "target"])

    # Normalize surrounding whitespace.
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()

    # Exclude empty translation pairs.
    df = df[
        (df["source"] != "") &
        (df["target"] != "")
    ]

    # Convert to a set for efficient duplicate detection.
    return set(zip(df["source"], df["target"]))

# Load each dataset.
datasets = {
    name: load_pairs(path)
    for name, path in paths.items()
}

# Display the number of unique translation pairs.
for name, pairs in datasets.items():
    print(f"{name}: {len(pairs):,} unique pairs")

# Combine training, validation and test pairs.
train = datasets["technical_train"] | datasets["general_train"]
validation = (
    datasets["technical_validation"] |
    datasets["general_validation"]
)
test = datasets["technical_test"] | datasets["general_test"]

# Detect exact translation-pair overlap between splits.
print("\n--- Dataset leakage checks ---")
print("Train / validation overlap:", len(train & validation))
print("Train / test overlap:", len(train & test))
print("Validation / test overlap:", len(validation & test))

# Stop if exact-pair leakage is detected.
if train & test or validation & test:
    raise ValueError(
        "Held-out test leakage detected. "
        "Resolve overlapping pairs before training E2."
    )

print("\nNo exact-pair test leakage detected.")
