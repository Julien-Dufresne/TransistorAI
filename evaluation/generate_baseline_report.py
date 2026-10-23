
"""
TransistorAI - Day 5 Baseline Report Generator

Read the completed benchmark results and generate
a reproducible Markdown report.

Input:
    results/day5_scores.csv

Output:
    results/day5_baseline_report.md
"""

import json
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results"
SCORES_PATH = RESULTS_DIR / "day5_scores.csv"
REPORT_PATH = RESULTS_DIR / "day5_baseline_report.md"


# --------------------------------------------------
# 2. LOAD BENCHMARK RESULTS
# --------------------------------------------------

if not SCORES_PATH.exists():
    raise FileNotFoundError(
        f"Benchmark scores not found: {SCORES_PATH}. "
        "Run python -m src.benchmark first."
    )

df = pd.read_csv(SCORES_PATH)

if df.empty:
    raise ValueError("No benchmark scores were produced.")


# --------------------------------------------------
# 3. VALIDATE REQUIRED COLUMNS
# --------------------------------------------------

required_columns = [
    "language_pair",
    "domain",
    "bleu",
    "chrf",
    "sentences_per_second",
    "model_id",
    "decoding_config",
]

missing = set(required_columns) - set(df.columns)

if missing:
    raise ValueError(
        f"Missing benchmark columns: {sorted(missing)}"
    )


# --------------------------------------------------
# 4. BUILD RESULTS TABLE
# --------------------------------------------------

table_columns = [
    "language_pair",
    "domain",
    "bleu",
    "chrf",
    "sentences_per_second",
]

# Include optional columns when available.
if "num_sentences" in df.columns:
    table_columns.insert(2, "num_sentences")

table = df[table_columns].round(2).to_markdown(
    index=False
)


# --------------------------------------------------
# 5. COLLECT EXPERIMENT CONFIGURATION
# --------------------------------------------------

configurations = []

for _, row in df.iterrows():

    configuration = {
        "language_pair": row["language_pair"],
        "domain": row["domain"],
        "model_id": row["model_id"],
        "decoding": json.loads(row["decoding_config"]),
    }

    for column in [
        "device",
        "batch_size",
        "dataset",
        "bleu_signature",
        "chrf_signature",
        "sacrebleu_version",
    ]:
        if column in df.columns:
            value = row[column]
            if pd.notna(value):
                configuration[column] = value

    configurations.append(configuration)

config_json = json.dumps(
    configurations,
    indent=2,
    ensure_ascii=False,
)


# --------------------------------------------------
# 6. GENERATE MARKDOWN REPORT
# --------------------------------------------------

report = f"""# TransistorAI — Day 5 Baseline Benchmark

## Research objective

Evaluate pretrained MarianMT translation models
before domain-specific fine-tuning.

The objective is to establish reproducible baseline
translation quality across technical and general domains.

## Experimental results

{table}

## Evaluation methodology

- Corpus-level SacreBLEU BLEU and chrF
- Held-out technical and general-domain test sets
- Fixed decoding configuration
- One reference translation per source sentence
- Batched inference using PyTorch

## Metric and model configuration

```json
{config_json}
```

## Interpretation

Compare technical and general-domain performance
within each language direction.

Higher BLEU and chrF scores generally indicate
greater agreement with reference translations.

Differences between domains may reflect differences
in vocabulary, sentence complexity and test data.

Avoid directly comparing BLEU scores across language
directions as though they represent identical tasks.

## Limitations

BLEU and chrF do not fully measure:

- Semantic accuracy
- Technical terminology correctness
- Hallucinations
- Omissions
- Contextual appropriateness

## Next steps

1. Inspect low-quality translations manually.
2. Analyze terminology and semantic errors.
3. Prepare domain-specific fine-tuning experiments.
4. Compare fine-tuned models against this baseline.
"""


# --------------------------------------------------
# 7. SAVE REPORT
# --------------------------------------------------

REPORT_PATH.write_text(
    report,
    encoding="utf-8",
)

print(f"Baseline report saved: {REPORT_PATH}")
print(f"Included {len(df)} benchmark results.")
