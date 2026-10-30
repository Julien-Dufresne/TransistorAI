
# ============================================================
# TransistorAI — Day 9: Qualitative Error Analysis
#
# Purpose:
# Compare E1 (technical fine-tuning) against E2 (mixed).
# Generate a reproducible sample for manual review.
#
# Run:
# python -m evaluation.qualitative_error_analysis
# ============================================================

from pathlib import Path
import json
import pandas as pd

# ------------------------------------------------------------
# 1. Define project paths and experiment settings
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/day8-de-en-v2"
OUTPUT = ROOT / "results/day9_error_analysis"

E1_PATH = BASE / "e1_evaluation/predictions.csv"
E2_PATH = BASE / "e2_evaluation/predictions.csv"

SEED = 42
SAMPLE_PER_DOMAIN = 25

# Use 20 model disagreements and 5 random controls per domain
# when enough examples are available.
DISAGREEMENT_TARGET = 20

KEYS = [
    "language_pair",
    "domain",
    "dataset",
    "sentence_id",
]

OUTPUT.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# 2. Load and validate the prediction files
# ------------------------------------------------------------

def load_predictions(path, model):
    """Load predictions and verify required columns."""

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path, keep_default_na=False)

    required = set(KEYS + [
        "source", "reference", "prediction"
    ])

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{model} is missing columns: {missing}"
        )

    # Ensure each sentence appears only once per experiment.
    if df.duplicated(KEYS).any():
        raise ValueError(
            f"{model} contains duplicate sentence keys."
        )

    # Check expected test-set sizes.
    expected = {"technical": 12623, "general": 1997}
    counts = df["domain"].value_counts().to_dict()

    if counts != expected:
        raise ValueError(
            f"{model}: unexpected domain counts: {counts}"
        )

    if (df["language_pair"] != "de-en").any():
        raise ValueError(
            f"{model} contains unexpected language pairs."
        )

    # Give each model's translation its own column.
    return df[
        KEYS + ["source", "reference", "prediction"]
    ].rename(
        columns={"prediction": f"{model}_prediction"}
    )


e1 = load_predictions(E1_PATH, "e1")
e2 = load_predictions(E2_PATH, "e2")

print("E1 and E2 prediction files validated.")


# ------------------------------------------------------------
# 3. Match the same sentences across both experiments
# ------------------------------------------------------------

merged = e1.merge(
    e2,
    on=KEYS,
    how="outer",
    validate="one_to_one",
    indicator=True,
    suffixes=("_e1", "_e2"),
)

# Stop if any sentence is missing from either experiment.
if not (merged["_merge"] == "both").all():
    raise ValueError(
        "E1 and E2 do not contain identical sentence keys."
    )

# Verify original source and reference translations.
for column in ["source", "reference"]:

    if not (
        merged[f"{column}_e1"]
        == merged[f"{column}_e2"]
    ).all():
        raise ValueError(
            f"Mismatch detected in {column}."
        )

merged["source"] = merged["source_e1"]
merged["reference"] = merged["reference_e1"]

merged = merged[
    KEYS + [
        "source",
        "reference",
        "e1_prediction",
        "e2_prediction",
    ]
].copy()

# Identify sentences where model outputs differ.
merged["models_disagree"] = (
    merged["e1_prediction"]
    != merged["e2_prediction"]
)

print(f"Matched sentences: {len(merged):,}")


# ------------------------------------------------------------
# 4. Sample technical and general examples
#
# Prefer disagreements because they are useful for comparing
# model behavior. Include random controls to reduce selection
# bias. This sample is NOT a representative error-rate estimate.
# ------------------------------------------------------------

samples = []
manifest = {
    "seed": SEED,
    "sample_per_domain": SAMPLE_PER_DOMAIN,
    "disagreement_target_per_domain": DISAGREEMENT_TARGET,
    "sampling_method": (
        "Disagreement-enriched sampling with random controls"
    ),
    "domains": {},
}

for domain in ["technical", "general"]:

    subset = merged[
        merged["domain"] == domain
    ].copy()

    disagree = subset[
        subset["models_disagree"]
    ]

    # Select up to 20 differing translations.
    n_disagree = min(
        DISAGREEMENT_TARGET,
        len(disagree),
        SAMPLE_PER_DOMAIN,
    )

    selected_disagree = disagree.sample(
        n=n_disagree,
        random_state=SEED,
    ).copy()

    selected_disagree["sampling_group"] = "disagreement"

    # Fill the remaining sample with random controls.
    remaining = subset.drop(
        index=selected_disagree.index
    )

    n_controls = SAMPLE_PER_DOMAIN - n_disagree

    selected_controls = remaining.sample(
        n=n_controls,
        random_state=SEED + 1,
    ).copy()

    selected_controls["sampling_group"] = "random_control"

    selected = pd.concat([
        selected_disagree,
        selected_controls,
    ])

    samples.append(selected)

    manifest["domains"][domain] = {
        "population": len(subset),
        "disagreements_in_population": len(disagree),
        "selected_disagreements": n_disagree,
        "selected_random_controls": n_controls,
    }

review = pd.concat(
    samples,
    ignore_index=True,
)

# Shuffle the final sample reproducibly.
review = review.sample(
    frac=1,
    random_state=SEED,
).reset_index(drop=True)

review.insert(
    0,
    "review_id",
    range(1, len(review) + 1),
)


# ------------------------------------------------------------
# 5. Create manual annotation columns
#
# These fields are intentionally empty.
# Human evaluation must be performed separately.
# ------------------------------------------------------------

annotation_columns = [
    "e1_error_category",
    "e2_error_category",
    "e1_severity",
    "e2_severity",
    "e1_meaning_preserved",
    "e2_meaning_preserved",
    "preferred_model",
    "comments",
]

for column in annotation_columns:
    review[column] = ""

# Organize columns for easier spreadsheet review.
review = review[
    [
        "review_id",
        "domain",
        "sentence_id",
        "sampling_group",
        "models_disagree",
        "source",
        "reference",
        "e1_prediction",
        "e2_prediction",
    ] + annotation_columns
]


# ------------------------------------------------------------
# 6. Save the review file and sampling manifest
# ------------------------------------------------------------

review_path = OUTPUT / "manual_review.csv"
manifest_path = OUTPUT / "sampling_manifest.json"

# UTF-8 BOM helps Excel recognize special characters.
review.to_csv(
    review_path,
    index=False,
    encoding="utf-8-sig",
)

manifest_path.write_text(
    json.dumps(manifest, indent=2),
    encoding="utf-8",
)

print("\nQualitative review dataset created.")
print(f"Total examples: {len(review)}")
print(review["domain"].value_counts())
print(f"\nReview file: {review_path}")
print(f"Sampling manifest: {manifest_path}")
