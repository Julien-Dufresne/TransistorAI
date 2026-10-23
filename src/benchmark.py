
"""
TransistorAI - Day 5 Baseline Benchmark

Evaluate pretrained MarianMT translation models on
available held-out technical and general-domain datasets.

Outputs:
    results/day5_predictions.csv
    results/day5_scores.csv
    results/day5_baseline_report.md (generated separately)
"""

import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from evaluation.metrics import calculate_metrics


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results"
CONFIG_PATH = ROOT / "evaluation" / "metric_config.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. LOAD CONFIGURATION
# --------------------------------------------------

with CONFIG_PATH.open(encoding="utf-8") as file:
    CONFIG = json.load(file)

MODELS = CONFIG["models"]
DECODING = CONFIG["decoding"]
BATCH_SIZE = CONFIG["batch_size"]

SOURCE_COLUMN = CONFIG["source_column"]
REFERENCE_COLUMN = CONFIG["reference_column"]


# --------------------------------------------------
# 3. SELECT HARDWARE
# --------------------------------------------------

# CUDA is used on compatible NVIDIA GPUs.
# MPS accelerates supported operations on Apple Silicon.
# CPU is the fallback.
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

print(f"Benchmark device: {DEVICE}")


# --------------------------------------------------
# 4. FIND TEST DATASETS
# --------------------------------------------------

def discover_datasets():
    """
    Discover held-out CSV test datasets.

    Expected layout:
        data/processed/<domain>/<direction>/*test.csv

    For example:
        data/processed/general/de-en/general_test.csv
    """

    datasets = []

    for path in sorted(DATA_DIR.rglob("*test.csv")):

        relative = path.relative_to(DATA_DIR)
        parts = relative.parts

        # Require a domain and language direction.
        if len(parts) < 3:
            continue

        domain, direction = parts[0], parts[1]

        # Only benchmark known model directions.
        if direction not in MODELS:
            continue

        if domain not in ("general", "technical"):
            continue

        datasets.append({
            "path": path,
            "domain": domain,
            "direction": direction,
        })

    return datasets


# --------------------------------------------------
# 5. TRANSLATION INFERENCE
# --------------------------------------------------

def translate_batch(texts, tokenizer, model):
    """
    Translate a batch using fixed decoding parameters.

    Using the same decoding configuration across
    experiments makes model comparisons more reliable.
    """

    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(DEVICE)

    # No gradients are needed during inference.
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            **DECODING,
        )

    return tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )


# --------------------------------------------------
# 6. RUN BASELINE EVALUATION
# --------------------------------------------------

def run_benchmark():

    datasets = discover_datasets()

    if not datasets:
        raise FileNotFoundError(
            "No test datasets found under data/processed."
        )

    all_predictions = []
    all_scores = []

    # Group datasets by translation direction.
    # This allows each model to be loaded only once.
    for direction, model_id in MODELS.items():

        selected = [
            dataset for dataset in datasets
            if dataset["direction"] == direction
        ]

        if not selected:
            continue

        print(f"\nLoading model: {model_id}")

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_id
        ).to(DEVICE)

        # Disable dropout and other training behavior.
        model.eval()

        for dataset in selected:

            path = dataset["path"]
            domain = dataset["domain"]

            print(f"\nEvaluating: {direction} / {domain}")

            df = pd.read_csv(path)

            # Validate the expected columns.
            required = {SOURCE_COLUMN, REFERENCE_COLUMN}

            if not required.issubset(df.columns):
                raise ValueError(
                    f"{path}: expected columns {required}"
                )

            if df[[SOURCE_COLUMN, REFERENCE_COLUMN]].isna().any().any():
                raise ValueError(
                    f"{path}: null translation values found"
                )

            sources = df[SOURCE_COLUMN].astype(str).tolist()
            references = df[REFERENCE_COLUMN].astype(str).tolist()

            if not sources:
                raise ValueError(f"{path}: empty test set")

            predictions = []

            start_time = time.perf_counter()

            # Translate the dataset in small batches.
            for start in range(0, len(sources), BATCH_SIZE):

                batch = sources[start:start + BATCH_SIZE]

                translated = translate_batch(
                    batch, tokenizer, model
                )

                predictions.extend(translated)

            elapsed = time.perf_counter() - start_time

            # Compute corpus BLEU and chrF.
            scores = calculate_metrics(
                predictions,
                references,
            )

            # Preserve the exact source/reference/prediction
            # alignment for later manual error analysis.
            for i, (source, reference, prediction) in enumerate(
                zip(sources, references, predictions)
            ):
                all_predictions.append({
                    "language_pair": direction,
                    "domain": domain,
                    "dataset": str(path.relative_to(ROOT)),
                    "model_id": model_id,
                    "sentence_id": i,
                    "source": source,
                    "reference": reference,
                    "prediction": prediction,
                })

            # Store aggregate results per test set.
            all_scores.append({
                "language_pair": direction,
                "domain": domain,
                "dataset": str(path.relative_to(ROOT)),
                "model_id": model_id,
                "device": DEVICE,
                "batch_size": BATCH_SIZE,
                "decoding_config": json.dumps(DECODING),
                "elapsed_seconds": round(elapsed, 3),
                "sentences_per_second": round(
                    len(sources) / elapsed, 3
                ),
                **scores,
            })

            print(
                f"BLEU: {scores['bleu']:.2f} | "
                f"chrF: {scores['chrf']:.2f}"
            )

        # Release memory before the next model.
        del model
        del tokenizer

        if DEVICE == "mps":
            torch.mps.empty_cache()

    # --------------------------------------------------
    # 7. SAVE OUTPUTS
    # --------------------------------------------------

    pd.DataFrame(all_predictions).to_csv(
        RESULTS_DIR / "day5_predictions.csv",
        index=False,
    )

    pd.DataFrame(all_scores).to_csv(
        RESULTS_DIR / "day5_scores.csv",
        index=False,
    )

    print("\nBenchmark completed.")
    print(f"Evaluated {len(all_scores)} datasets.")

    return all_scores


if __name__ == "__main__":
    run_benchmark()
