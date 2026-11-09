
"""
Day 18 — E0–E3 translation smoke tests.

Checks whether all four translation systems can load and
generate translations from a small fixed set of German inputs.

Results are saved as CSV for later analysis.

Run:
    HF_HUB_OFFLINE=1 python evaluation/day18_translation_smoke.py
"""

import csv
import os
import sys
import time
from pathlib import Path

# Resolve project paths independently of the working directory.
ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results/day18_smoke_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Prevent Hugging Face Hub downloads during the test.
# Set these before importing model libraries.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Model configurations match the Streamlit MVP.
MODELS = {
    "E0": "Helsinki-NLP/opus-mt-de-en",
    "E1": str(
        ROOT / "results/day8-de-en-v2/technical-finetuned/final_model"
    ),
    "E2": str(
        ROOT / "results/day8-de-en-v2/mixed-finetuned/final_model"
    ),
    "E3": "mlx-community/Qwen2.5-3B-Instruct-4bit",
}

# Use the same deterministic decoding parameters as the MVP.
DECODING_CONFIG = {
    "num_beams": 4,
    "do_sample": False,
    "max_new_tokens": 128,
}

# Use Apple Silicon acceleration when available.
DEVICE = (
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

# Fixed examples cover important translation behaviors.
TEST_CASES = [
    {
        "case": "general",
        "source": "Heute ist das Wetter sehr schön.",
    },
    {
        "case": "technical",
        "source": "Starten Sie den Server neu.",
    },
    {
        "case": "negation",
        "source": "Löschen Sie diese Datei nicht.",
    },
    {
        "case": "numbers",
        "source": "Der Wert beträgt 12,5 Prozent.",
    },
    {
        "case": "placeholder",
        "source": "Öffnen Sie die Datei {filename}.",
    },
]


def synchronize_device():
    """Wait for outstanding GPU operations before timing."""
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def load_nmt(model_path):
    """Load a locally cached NMT model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        local_files_only=True,
    )

    model = model.to(DEVICE)
    model.eval()

    return model, tokenizer


def translate_nmt(model, tokenizer, source):
    """Generate one deterministic NMT translation."""
    inputs = tokenizer(
        source,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    synchronize_device()
    start = time.perf_counter()

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            **DECODING_CONFIG,
        )

    synchronize_device()
    latency_ms = (time.perf_counter() - start) * 1000

    translation = tokenizer.decode(
        generated[0],
        skip_special_tokens=True,
    )

    return translation.strip(), latency_ms


def load_qwen():
    """Load Qwen using the local MLX model cache."""
    from mlx_lm import load

    return load(MODELS["E3"])


def translate_qwen(model, tokenizer, source):
    """Generate one deterministic Qwen translation."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional German-to-English translator. "
                "Preserve terminology, numbers, placeholders, and negation. "
                "Return only the English translation."
            ),
        },
        {
            "role": "user",
            "content": source,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    start = time.perf_counter()
    parts = []

    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=128,
        sampler=make_sampler(temp=0),
    ):
        parts.append(response.text)

    latency_ms = (time.perf_counter() - start) * 1000

    return "".join(parts).strip(), latency_ms


def run_smoke_tests():
    """Execute smoke tests and save one result per model and case."""
    results = []

    print(f"Device: {DEVICE}", flush=True)

    for model_key, model_path in MODELS.items():
        print(f"\nLoading {model_key}...", flush=True)

        # Model loading is separated from translation timing.
        try:
            if model_key == "E3":
                model, tokenizer = load_qwen()
            else:
                model, tokenizer = load_nmt(model_path)

        except Exception as exc:
            print(f"LOAD FAILED: {model_key}: {exc}", flush=True)

            # Record each skipped case so failures remain visible.
            for case in TEST_CASES:
                results.append({
                    "model": model_key,
                    "case": case["case"],
                    "source": case["source"],
                    "translation": "",
                    "latency_ms": "",
                    "status": "LOAD_FAILED",
                    "error": str(exc),
                })

            continue

        for case in TEST_CASES:
            try:
                if model_key == "E3":
                    translation, latency = translate_qwen(
                        model, tokenizer, case["source"]
                    )
                else:
                    translation, latency = translate_nmt(
                        model, tokenizer, case["source"]
                    )

                # A smoke test checks execution, not translation quality.
                status = "PASS" if translation else "EMPTY_OUTPUT"
                error = ""

            except Exception as exc:
                translation = ""
                latency = ""
                status = "INFERENCE_FAILED"
                error = str(exc)

            results.append({
                "model": model_key,
                "case": case["case"],
                "source": case["source"],
                "translation": translation,
                "latency_ms": latency,
                "status": status,
                "error": error,
            })

            print(
                f"{model_key} | {case['case']} | {status} | "
                f"{translation}",
                flush=True,
            )

        # Release the model before loading the next checkpoint.
        del model
        del tokenizer

        if DEVICE == "cuda":
            torch.cuda.empty_cache()
        elif DEVICE == "mps":
            torch.mps.empty_cache()

    output_path = OUTPUT_DIR / "smoke_results.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "model", "case", "source", "translation",
                "latency_ms", "status", "error",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    passed = sum(row["status"] == "PASS" for row in results)

    print(f"\nResults saved to: {output_path}")
    print(f"Passed: {passed}/{len(results)}")

    # Return a nonzero exit code if any smoke test failed.
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(run_smoke_tests())
