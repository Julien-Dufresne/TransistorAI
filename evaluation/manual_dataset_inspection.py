
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "evaluation"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "train": DATA_DIR / "technical/de-en/train.csv",
    "validation": DATA_DIR / "technical/de-en/validation.csv",
    "technical_test": DATA_DIR / "technical/de-en/technical_test.csv",
    "general_test": DATA_DIR / "general/de-en/general_test.csv",
}

SAMPLE_SIZES = {
    "train": 40,
    "validation": 30,
    "technical_test": 30,
    "general_test": 30,
}

samples = []

for name, path in DATASETS.items():
    df = pd.read_csv(path)

    # Keep the original CSV row for traceability.
    df = df.reset_index().rename(
        columns={"index": "original_row"}
    )

    sample = df.sample(
        n=min(SAMPLE_SIZES[name], len(df)),
        random_state=42
    ).copy()

    sample["dataset"] = name
    sample["pair_id"] = [
        f"{name.upper()}_{i:03d}"
        for i in range(1, len(sample) + 1)
    ]

    samples.append(sample)

inspection = pd.concat(samples, ignore_index=True)

# Fields to complete manually
inspection["language_ok"] = ""
inspection["alignment_ok"] = ""
inspection["meaning_score"] = ""
inspection["fluency_score"] = ""
inspection["terminology_score"] = ""
inspection["error_category"] = ""
inspection["notes"] = ""

columns = [
    "pair_id",
    "dataset",
    "original_row",
    "source",
    "target",
    "language_ok",
    "alignment_ok",
    "meaning_score",
    "fluency_score",
    "terminology_score",
    "error_category",
    "notes",
]

inspection = inspection[columns]

output_path = OUTPUT_DIR / "manual_dataset_inspection.csv"

inspection.to_csv(output_path, index=False)

print(f"Saved {len(inspection)} sentence pairs")
print(f"File: {output_path}")
print(inspection.groupby("dataset").size())
