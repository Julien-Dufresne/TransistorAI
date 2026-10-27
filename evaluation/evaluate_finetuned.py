
"""
TransistorAI — Day 7 Fine-Tuned Model Evaluation.

Evaluate the fine-tuned MarianMT model on the same held-out
technical and general-domain datasets used on Day 5.

Outputs:
    results/day7_predictions.csv
    results/day7_scores.csv
    results/day7_evaluation_report.md
"""

import json
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Reuse the exact SacreBLEU implementation from Day 5.
from evaluation.metrics import calculate_metrics


# STEP 1 — Define project paths.
ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT / "configs" / "day7_eval_config.json"
DAY5_CONFIG_PATH = ROOT / "configs" / "metric_config.json"


def select_device():
    """Select the best available inference device."""

    # Use NVIDIA GPU acceleration when available.
    if torch.cuda.is_available():
        return "cuda"

    # Use Apple Silicon GPU acceleration when available.
    if torch.backends.mps.is_available():
        return "mps"

    # Fall back to CPU.
    return "cpu"


def translate_batch(texts, tokenizer, model, device, decoding):
    """Generate translations using Day 5 inference behavior."""

    # Tokenize source sentences exactly as in Day 5.
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    # Disable gradient calculation during inference.
    with torch.inference_mode():

        # Generate translations using the fixed decoding settings.
        output_ids = model.generate(
            **inputs,
            **decoding,
        )

    # Convert generated token IDs back into text.
    return tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )


def main():
    """Evaluate the fine-tuned model and export results."""

    # STEP 2 — Load the Day 7 evaluation configuration.
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = json.load(file)

    # STEP 3 — Load the original Day 5 metric configuration.
    with DAY5_CONFIG_PATH.open(encoding="utf-8") as file:
        day5_config = json.load(file)

    # Extract the decoding settings and batch size used on Day 5.
    day5_decoding = day5_config["decoding"]
    day5_batch_size = day5_config["batch_size"]

    # Extract the corresponding Day 7 settings.
    day7_decoding = {
        key: value
        for key, value in config["decoding"].items()
        if key != "batch_size"
    }

    batch_size = config["decoding"]["batch_size"]

    # STEP 4 — Verify decoding consistency.
    # A fair comparison requires identical generation settings.
    if day7_decoding != day5_decoding:
        raise ValueError(
            "Day 7 decoding differs from Day 5. "
            f"Day 5: {day5_decoding}; Day 7: {day7_decoding}"
        )

    # Batch size can affect throughput and, in some circumstances,
    # generation behavior through padding.
    if batch_size != day5_batch_size:
        raise ValueError(
            "Day 7 batch size differs from Day 5."
        )

    # Verify that source/reference columns match Day 5.
    if (
        config["source_column"] != day5_config["source_column"]
        or config["target_column"] != day5_config["reference_column"]
    ):
        raise ValueError(
            "Day 7 text columns differ from Day 5."
        )

    # STEP 5 — Select the inference device.
    device = select_device()

    print(f"Evaluation device: {device}")

    # STEP 6 — Load the fine-tuned model and tokenizer.
    model_path = ROOT / config["model_path"]

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Fine-tuned model not found: {model_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path
    ).to(device)

    # Disable dropout and other training-specific behavior.
    model.eval()

    # STEP 7 — Prepare result containers.
    all_predictions = []
    all_scores = []

    language_pair = config["language_pair"]

    # STEP 8 — Evaluate each held-out test dataset.
    for domain, relative_path in config["datasets"].items():

        dataset_path = ROOT / relative_path

        print(f"\nEvaluating {language_pair} / {domain}")

        # Load the original held-out test data.
        df = pd.read_csv(dataset_path)

        source_column = config["source_column"]
        target_column = config["target_column"]

        # Verify the expected columns exist.
        required = {source_column, target_column}

        if not required.issubset(df.columns):
            raise ValueError(
                f"Missing required columns in {dataset_path}"
            )

        # Match the Day 5 null-value validation.
        if df[[source_column, target_column]].isna().any().any():
            raise ValueError(
                f"Null translation values in {dataset_path}"
            )

        # Preserve the original sentence order.
        sources = df[source_column].astype(str).tolist()
        references = df[target_column].astype(str).tolist()

        if not sources:
            raise ValueError(
                f"Empty evaluation dataset: {dataset_path}"
            )

        predictions = []

        # Start measuring translation inference time.
        start_time = time.perf_counter()

        # STEP 9 — Generate translations in batches.
        for start in range(0, len(sources), batch_size):

            batch = sources[start:start + batch_size]

            translated = translate_batch(
                batch,
                tokenizer,
                model,
                device,
                day7_decoding,
            )

            predictions.extend(translated)

        # Calculate total inference time.
        elapsed = time.perf_counter() - start_time

        # Verify that every sentence received a translation.
        if len(predictions) != len(references):
            raise ValueError(
                "Prediction/reference count mismatch."
            )

        # STEP 10 — Calculate BLEU and chrF.
        # This is the same function used by the Day 5 benchmark.
        scores = calculate_metrics(
            predictions,
            references,
        )

        # STEP 11 — Store sentence-level predictions.
        for i, (source, reference, prediction) in enumerate(
            zip(sources, references, predictions)
        ):
            all_predictions.append({
                "language_pair": language_pair,
                "domain": domain,
                "dataset": relative_path,
                "model_id": config["model_path"],
                "sentence_id": i,
                "source": source,
                "reference": reference,
                "prediction": prediction,
            })

        # STEP 12 — Store corpus-level evaluation scores.
        all_scores.append({
            "language_pair": language_pair,
            "domain": domain,
            "dataset": relative_path,
            "model_id": config["model_path"],
            "device": device,
            "batch_size": batch_size,
            "decoding_config": json.dumps(day7_decoding),
            "elapsed_seconds": round(elapsed, 3),
            "sentences_per_second": round(
                len(sources) / elapsed, 3
            ),
            **scores,
        })

        print(
            f"BLEU: {scores['bleu']:.4f} | "
            f"chrF: {scores['chrf']:.4f}"
        )

    # STEP 13 — Export evaluation results.
    predictions_path = ROOT / config["output"]["predictions"]
    scores_path = ROOT / config["output"]["scores"]

    # Create output directories if necessary.
    predictions_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    scores_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save individual translations.
    pd.DataFrame(all_predictions).to_csv(
        predictions_path,
        index=False,
    )

    # Save corpus-level BLEU and chrF scores.
    pd.DataFrame(all_scores).to_csv(
        scores_path,
        index=False,
    )

    # STEP 14 — Generate a Markdown evaluation report.
    report_path = ROOT / config["output"]["report"]

    report_lines = [
        "# Day 7 — Fine-Tuned Model Evaluation",
        "",
        f"**Experiment:** {config['experiment_name']}",
        f"**Model:** {config['model_path']}",
        f"**Language pair:** {language_pair}",
        f"**Device:** {device}",
        "",
        "## Evaluation configuration",
        "",
        f"- Batch size: {batch_size}",
        f"- Decoding: `{json.dumps(day7_decoding)}`",
        "- Metrics: SacreBLEU BLEU and chrF (Day 5 implementation)",
        "",
        "## Evaluation results",
        "",
        "| Domain | Sentences | BLEU | chrF | Sentences/sec |",
        "|---|---:|---:|---:|---:|",
    ]

    # Add one report row per domain.
    for row in all_scores:
        report_lines.append(
            f"| {row['domain']} "
            f"| {row['sentence_count']} "
            f"| {row['bleu']:.4f} "
            f"| {row['chrf']:.4f} "
            f"| {row['sentences_per_second']:.3f} |"
        )

    report_lines.extend([
        "",
        "## Reproducibility",
        "",
        f"- SacreBLEU version: {all_scores[0]['sacrebleu_version']}",
        f"- BLEU signature: `{all_scores[0]['bleu_signature']}`",
        f"- chrF signature: `{all_scores[0]['chrf_signature']}`",
        "",
        "The evaluation reuses the Day 5 metric implementation, "
        "tokenization behavior, decoding parameters and batch size.",
    ])

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    # STEP 15 — Confirm successful export.
    print("\nDay 7 evaluation completed.")
    print(f"Predictions saved: {predictions_path}")
    print(f"Scores saved: {scores_path}")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
