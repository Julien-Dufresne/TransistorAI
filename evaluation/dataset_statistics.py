
from pathlib import Path
import pandas as pd
import json

# STEP 1 - Define paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# STEP 2 - Define datasets
datasets = {
    "train": DATA_DIR / "technical/de-en/train.csv",
    "validation": DATA_DIR / "technical/de-en/validation.csv",
    "technical_test": DATA_DIR / "technical/de-en/technical_test.csv",
    "general_test": DATA_DIR / "general/de-en/general_test.csv",
}

# STEP 3 - Calculate statistics
statistics = {}

for name, path in datasets.items():

    df = pd.read_csv(path)

    source_lengths = df["source"].str.split().str.len()
    target_lengths = df["target"].str.split().str.len()

    stats = {
        "sentence_pairs": len(df),
        "avg_source_words": round(source_lengths.mean(), 2),
        "avg_target_words": round(target_lengths.mean(), 2),
        "max_source_words": int(source_lengths.max()),
        "max_target_words": int(target_lengths.max()),
        "duplicate_pairs": int(
            df.duplicated(["source", "target"]).sum()
        ),
    }

    statistics[name] = stats

    print(f"\n{name.upper()}")
    for key, value in stats.items():
        print(f"{key}: {value}")

# STEP 4 - Save results
output_path = RESULTS_DIR / "day4_dataset_statistics.json"

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(statistics, f, indent=4, ensure_ascii=False)

print(f"\nStatistics saved to: {output_path}")
