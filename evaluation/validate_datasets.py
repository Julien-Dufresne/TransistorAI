
# ============================================================
# TransistorAI - Dataset Validation
# Day 4: Technical and General-Domain Data
# ============================================================

from pathlib import Path
import pandas as pd

# STEP 1 - Define project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

DATASETS = {
    "train": DATA_DIR / "technical/de-en/train.csv",
    "validation": DATA_DIR / "technical/de-en/validation.csv",
    "technical_test": DATA_DIR / "technical/de-en/technical_test.csv",
    "general_test": DATA_DIR / "general/de-en/general_test.csv",
}

# STEP 2 - Load and validate each dataset
datasets = {}

for name, path in DATASETS.items():

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    df = pd.read_csv(path)

    print(f"\n--- {name.upper()} ---")

    # Check required columns
    assert {"source", "target"}.issubset(df.columns), (
        f"{name}: Missing source or target column"
    )

    # Check missing and empty translations
    assert not df[["source", "target"]].isna().any().any(), (
        f"{name}: Missing values detected"
    )

    assert not (
        df["source"].astype(str).str.strip().eq("")
        | df["target"].astype(str).str.strip().eq("")
    ).any(), f"{name}: Empty sentences detected"

    # Check exact duplicate sentence pairs
    duplicates = df.duplicated(
        subset=["source", "target"]
    ).sum()

    assert duplicates == 0, (
        f"{name}: {duplicates} duplicate pairs detected"
    )

    print(f"Sentence pairs: {len(df):,}")
    print(f"Duplicate pairs: {duplicates}")
    print("Missing values: 0")

    datasets[name] = df

# STEP 3 - Check source overlap across datasets
names = list(datasets.keys())

print("\n--- CROSS-DATASET OVERLAP ---")

for i, name_a in enumerate(names):
    for name_b in names[i + 1:]:

        sources_a = set(datasets[name_a]["source"])
        sources_b = set(datasets[name_b]["source"])

        overlap = sources_a.intersection(sources_b)

        print(
            f"{name_a} vs {name_b}: "
            f"{len(overlap)} shared source sentences"
        )

        assert len(overlap) == 0, (
            f"Data leakage detected between "
            f"{name_a} and {name_b}"
        )

# STEP 4 - Final confirmation
print("\nAll dataset validation checks passed!")
