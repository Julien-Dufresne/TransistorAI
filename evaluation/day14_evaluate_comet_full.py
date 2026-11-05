from pathlib import Path
import time

import pandas as pd
from comet import download_model, load_from_checkpoint

from day11_evaluate_comet import calculate_comet

BASE = Path("results/day14_llm_baseline")
PREDICTIONS = BASE / "predictions.csv"
SEGMENTS = BASE / "full_comet_segments.csv"
SUMMARY = BASE / "full_comet_scores.csv"

MODEL_NAME = "Unbabel/wmt22-comet-da"
BATCH_SIZE = 8


def main():
    df = pd.read_csv(PREDICTIONS, keep_default_na=False)

    required = [
        "language_pair", "domain", "dataset",
        "sentence_id", "source", "reference", "prediction"
    ]

    if not set(required).issubset(df.columns):
        raise ValueError("Missing required prediction columns")

    if len(df) != 14620:
        raise ValueError(f"Expected 14620 rows, found {len(df)}")

    if df.duplicated(
        ["language_pair", "domain", "dataset", "sentence_id"]
    ).any():
        raise ValueError("Duplicate sentence identifiers")

    if df[["source", "reference", "prediction"]].apply(
        lambda col: col.astype(str).str.strip().eq("")
    ).any().any():
        raise ValueError("Empty source, reference or prediction")

    print("Loading COMET model...", flush=True)
    model_path = download_model(MODEL_NAME)
    model = load_from_checkpoint(model_path)

    summaries = []
    segment_frames = []

    for domain in ["technical", "general"]:
        subset = df[df["domain"] == domain].copy()

        print(
            f"\nEvaluating {domain}: {len(subset)} sentences",
            flush=True
        )

        start = time.time()

        system_score, segment_scores = calculate_comet(
            model=model,
            df=subset,
            batch_size=BATCH_SIZE,
            gpus=0,
        )

        if len(segment_scores) != len(subset):
            raise ValueError("COMET segment count mismatch")

        subset["model"] = "E3_Qwen"
        subset["comet_score"] = segment_scores

        segment_frames.append(subset)

        elapsed = time.time() - start

        summaries.append({
            "model": "E3_Qwen",
            "domain": domain,
            "n": len(subset),
            "COMET": system_score,
            "runtime_seconds": elapsed,
        })

        # Save completed domains immediately.
        pd.concat(segment_frames, ignore_index=True).to_csv(
            SEGMENTS, index=False
        )

        pd.DataFrame(summaries).to_csv(
            SUMMARY, index=False
        )

        print(
            f"{domain}: COMET={system_score:.6f} "
            f"({elapsed:.1f} seconds)",
            flush=True
        )

    print("\nFULL E3 COMET RESULTS")
    print(pd.DataFrame(summaries).to_string(index=False))
    print("\nSaved:", SUMMARY)
    print("Saved:", SEGMENTS)


if __name__ == "__main__":
    main()
