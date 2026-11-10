"""
Day 17 — COMET evaluation of human preference sentences.

Scores the exact E0–E3 translations shown to human reviewers.
Reuses the existing Day 11 COMET implementation.

Input:
    results/day17_human_metric_comparison/sentence_metric_scores.csv

Output:
    results/day17_human_metric_comparison/sentence_metric_scores_comet.csv
"""

from pathlib import Path
import sys

import pandas as pd
from comet import download_model, load_from_checkpoint

# Allow imports from the evaluation directory.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))

from day11_evaluate_comet import calculate_comet


INPUT = (
    ROOT
    / "results/day17_human_metric_comparison/sentence_metric_scores.csv"
)

OUTPUT = (
    ROOT
    / "results/day17_human_metric_comparison/sentence_metric_scores_comet.csv"
)

MODEL_NAME = "Unbabel/wmt22-comet-da"

MODELS = ["E0", "E1", "E2", "E3"]


def main():
    print("\nDAY 17 — COMET EVALUATION\n", flush=True)

    # --------------------------------------------------------
    # 1. Load existing BLEU and chrF results
    # --------------------------------------------------------

    df = pd.read_csv(INPUT, keep_default_na=False)

    required_columns = {
        "evaluation_id",
        "domain",
        "model",
        "source",
        "reference",
        "prediction",
        "BLEU",
        "chrF",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.duplicated(["evaluation_id", "model"]).any():
        raise ValueError("Duplicate evaluation/model pairs detected.")

    if set(df["model"]) != set(MODELS):
        raise ValueError("Unexpected model IDs.")

    counts = df.groupby("evaluation_id")["model"].nunique()

    if not counts.eq(len(MODELS)).all():
        raise ValueError("Some sentences are missing model predictions.")

    if df[["source", "reference", "prediction"]].isna().any().any():
        raise ValueError("Missing source, reference or prediction.")

    print("Unique sentences:", df["evaluation_id"].nunique())
    print("Total predictions:", len(df))

    # --------------------------------------------------------
    # 2. Load COMET
    # --------------------------------------------------------

    print("\nLoading COMET model...", flush=True)

    checkpoint = download_model(MODEL_NAME)
    comet_model = load_from_checkpoint(checkpoint)

    print("COMET loaded successfully.", flush=True)

    # --------------------------------------------------------
    # 3. Evaluate each model separately
    # --------------------------------------------------------

    results = []

    for model_name in MODELS:

        subset = df[df["model"] == model_name].copy()

        print(
            f"\nEvaluating {model_name}: {len(subset)} sentences",
            flush=True,
        )

        system_score, segment_scores = calculate_comet(
            comet_model,
            subset,
            batch_size=8,
            gpus=0,
        )

        if len(segment_scores) != len(subset):
            raise ValueError(
                f"COMET returned {len(segment_scores)} scores "
                f"for {len(subset)} predictions."
            )

        subset["COMET"] = segment_scores

        results.append(subset)

        # Save progress after each model.
        partial = pd.concat(results, ignore_index=True)
        partial.to_csv(OUTPUT, index=False)

        print(
            f"{model_name} — COMET: {system_score:.4f}",
            flush=True,
        )

    # --------------------------------------------------------
    # 4. Final validation and summary
    # --------------------------------------------------------

    final = pd.concat(results, ignore_index=True)

    if len(final) != len(df):
        raise ValueError("Final row count mismatch.")

    if final["COMET"].isna().any():
        raise ValueError("Missing COMET scores.")

    print("\n=== OVERALL MODEL COMPARISON ===")

    print(
        final.groupby("model")[["BLEU", "chrF", "COMET"]]
        .mean()
        .round(4)
        .to_string()
    )

    print("\n=== DOMAIN COMPARISON ===")

    print(
        final.groupby(["domain", "model"])[
            ["BLEU", "chrF", "COMET"]
        ]
        .mean()
        .round(4)
        .to_string()
    )

    print("\nSaved:", OUTPUT)


if __name__ == "__main__":
    main()
