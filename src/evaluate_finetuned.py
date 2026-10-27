
# Day 7 — Evaluate the fine-tuned translation model.
#
# This script generates translations for both held-out
# test datasets and computes corpus-level BLEU and chrF.

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from sacrebleu.metrics import BLEU, CHRF


def select_device():
    """Select the best available inference device."""

    # Prefer CUDA, then Apple Silicon MPS, then CPU.
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(model_path, device):
    """Load the fine-tuned model and tokenizer."""

    # Load the tokenizer saved during training.
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    # Load the fine-tuned translation model.
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path
    )

    # Move the model to the selected device.
    model.to(device)

    # Disable training-specific behavior such as dropout.
    model.eval()

    return tokenizer, model


def generate_predictions(
    texts,
    tokenizer,
    model,
    device,
    decoding,
):
    """Translate source sentences in batches."""

    predictions = []

    # Read the inference batch size from configuration.
    batch_size = decoding["batch_size"]

    # Disable gradient calculations during inference.
    with torch.inference_mode():

        # Process sentences in manageable batches.
        for start in range(0, len(texts), batch_size):

            batch = texts[start:start + batch_size]

            # Tokenize the source sentences.
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128,
            )

            # Move tokenized inputs to the model device.
            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            # Generate translations using fixed decoding.
            generated = model.generate(
                **inputs,
                num_beams=decoding["num_beams"],
                do_sample=decoding["do_sample"],
                max_new_tokens=decoding["max_new_tokens"],
            )

            # Convert token IDs into readable text.
            translations = tokenizer.batch_decode(
                generated,
                skip_special_tokens=True,
            )

            # Store translations in original dataset order.
            predictions.extend(translations)

    return predictions


def evaluate_dataset(
    domain,
    dataset_path,
    tokenizer,
    model,
    device,
    config,
):
    """Evaluate one held-out test dataset."""

    # Load the parallel source-reference dataset.
    df = pd.read_csv(dataset_path)

    source_col = config["source_column"]
    target_col = config["target_column"]

    # Fail explicitly if required columns are missing.
    if not {source_col, target_col}.issubset(df.columns):
        raise ValueError(
            f"Missing columns in {dataset_path}"
        )

    # Test data must contain valid aligned pairs.
    if df[[source_col, target_col]].isna().any().any():
        raise ValueError("Test dataset contains null pairs.")

    # Preserve original sentence order and punctuation.
    sources = df[source_col].astype(str).tolist()
    references = df[target_col].astype(str).tolist()

    # Generate translations using the fine-tuned model.
    predictions = generate_predictions(
        sources,
        tokenizer,
        model,
        device,
        config["decoding"],
    )

    # Verify that every source received one translation.
    assert len(predictions) == len(references)

    # Compute corpus-level BLEU and chrF.
    # The reference corpus is wrapped in an outer list.
    bleu = BLEU().corpus_score(
        predictions,
        [references],
    ).score

    chrf = CHRF().corpus_score(
        predictions,
        [references],
    ).score

    # Store predictions for manual error analysis.
    prediction_rows = pd.DataFrame({
        "language_pair": config["language_pair"],
        "domain": domain,
        "dataset": dataset_path,
        "sentence_id": range(len(sources)),
        "source": sources,
        "reference": references,
        "prediction": predictions,
    })

    # Store one aggregate score row per dataset.
    score_row = {
        "language_pair": config["language_pair"],
        "domain": domain,
        "dataset": dataset_path,
        "model_type": "fine_tuned",
        "model_path": config["model_path"],
        "num_sentences": len(sources),
        "bleu": round(bleu, 4),
        "chrf": round(chrf, 4),
    }

    return prediction_rows, score_row


def main():
    """Run the complete Day 7 evaluation."""

    # STEP 1 — Load evaluation configuration.
    with open("configs/day7_eval_config.json") as file:
        config = json.load(file)

    # STEP 2 — Fix random seeds for reproducibility.
    seed = config["seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # STEP 3 — Select the inference device.
    device = select_device()
    print(f"Using device: {device}")

    # STEP 4 — Load the fine-tuned model.
    tokenizer, model = load_model(
        config["model_path"],
        device,
    )

    all_predictions = []
    all_scores = []

    # STEP 5 — Evaluate technical and general datasets.
    for domain, dataset_path in config["datasets"].items():

        print(f"Evaluating {domain} dataset...")

        predictions, scores = evaluate_dataset(
            domain,
            dataset_path,
            tokenizer,
            model,
            device,
            config,
        )

        all_predictions.append(predictions)
        all_scores.append(scores)

        print(
            f"{domain}: "
            f"BLEU={scores['bleu']:.4f}, "
            f"chrF={scores['chrf']:.4f}"
        )

    # STEP 6 — Create the results directory.
    Path("results").mkdir(
        parents=True,
        exist_ok=True,
    )

    # STEP 7 — Save sentence-level predictions.
    pd.concat(
        all_predictions,
        ignore_index=True,
    ).to_csv(
        config["output"]["predictions"],
        index=False,
    )

    # STEP 8 — Save corpus-level metric scores.
    pd.DataFrame(all_scores).to_csv(
        config["output"]["scores"],
        index=False,
    )

    print("Day 7 evaluation completed.")


if __name__ == "__main__":
    main()
