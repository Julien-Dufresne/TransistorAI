import argparse
import csv
import json
import time
from pathlib import Path

import pandas as pd
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

MODEL_ID = "mlx-community/Qwen2.5-3B-Instruct-4bit"
SOURCE_FILE = Path(
    "results/day11_evaluation/e0_generation/predictions.csv"
)
OUTPUT_DIR = Path("results/day14_llm_baseline")

SYSTEM_PROMPT = (
    "You are a professional German-to-English translator. "
    "Translate the German text into accurate, natural English. "
    "Preserve technical terminology, numbers, placeholders, and negation. "
    "Return only the English translation, without explanations."
)

KEYS = ["language_pair", "domain", "dataset", "sentence_id"]
TEXT_COLUMNS = ["source", "reference"]

OUTPUT_COLUMNS = [
    "language_pair", "domain", "dataset", "model_id",
    "sentence_id", "source", "reference", "prediction"
]

DIAGNOSTIC_COLUMNS = KEYS + [
    "model_id", "generation_tokens", "finish_reason",
    "potentially_truncated"
]

MAX_TOKENS = 128


def load_source(mode, per_domain):
    df = pd.read_csv(SOURCE_FILE, keep_default_na=False)

    required = set(OUTPUT_COLUMNS)
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing source columns: {sorted(missing)}")

    if df.duplicated(KEYS).any():
        raise ValueError("Duplicate identifiers in source dataset")

    for column in ["source", "reference"]:
        if df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Empty {column} in source dataset")

    if mode == "pilot":
        parts = []

        for domain in ["technical", "general"]:
            subset = df[df["domain"] == domain]

            if len(subset) < per_domain:
                raise ValueError(f"Not enough {domain} examples")

            parts.append(
                subset.sample(n=per_domain, random_state=42)
            )

        df = pd.concat(parts, ignore_index=True)

    df = df.drop(columns=["prediction"])
    df["model_id"] = MODEL_ID

    return df[OUTPUT_COLUMNS[:-1]]


def validate_checkpoint(path, source):
    if not path.exists():
        return set()

    existing = pd.read_csv(path, keep_default_na=False)

    missing = set(OUTPUT_COLUMNS) - set(existing.columns)
    if missing:
        raise ValueError(
            f"Checkpoint missing columns: {sorted(missing)}"
        )

    if existing.duplicated(KEYS).any():
        raise ValueError("Duplicate identifiers in checkpoint")

    expected = source.set_index(KEYS)
    completed = set()

    for row in existing.to_dict("records"):
        key = tuple(row[k] for k in KEYS)

        if key not in expected.index:
            raise ValueError(f"Unknown checkpoint identifier: {key}")

        original = expected.loc[key]

        if row["model_id"] != MODEL_ID:
            raise ValueError(f"Wrong model in checkpoint: {key}")

        for column in TEXT_COLUMNS:
            if row[column] != original[column]:
                raise ValueError(
                    f"Checkpoint {column} mismatch: {key}"
                )

        if not str(row["prediction"]).strip():
            raise ValueError(f"Empty checkpoint translation: {key}")

        completed.add(key)

    return completed


def generate_translation(model, tokenizer, sampler, source):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": source},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    chunks = []
    last_response = None

    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        sampler=sampler,
    ):
        chunks.append(response.text)
        last_response = response

    if last_response is None:
        raise RuntimeError("Generation returned no response")

    prediction = "".join(chunks).strip()

    if not prediction:
        raise ValueError("Empty translation generated")

    tokens = int(last_response.generation_tokens)
    reason = last_response.finish_reason

    truncated = (
        reason in {"length", "max_tokens"}
        or tokens >= MAX_TOKENS
    )

    return prediction, tokens, reason, truncated


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=["pilot", "full"],
        default="pilot",
    )

    parser.add_argument(
        "--per-domain",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of new translations this run",
    )

    args = parser.parse_args()

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    source = load_source(args.mode, args.per_domain)

    output_file = OUTPUT_DIR / (
        "pilot_predictions.csv"
        if args.mode == "pilot"
        else "predictions.csv"
    )

    diagnostic_file = OUTPUT_DIR / (
        "pilot_generation_diagnostics.csv"
        if args.mode == "pilot"
        else "generation_diagnostics.csv"
    )

    completed = validate_checkpoint(output_file, source)

    pending = [
        row for row in source.to_dict("records")
        if tuple(row[k] for k in KEYS) not in completed
    ]

    total_pending = len(pending)

    if args.limit is not None:
        pending = pending[:args.limit]

    print(f"Total examples: {len(source)}")
    print(f"Previously completed: {len(completed)}")
    print(f"Remaining before limit: {total_pending}")
    print(f"Generating this run: {len(pending)}", flush=True)

    if not pending:
        print("No new translations required.")
        return

    print(f"Loading {MODEL_ID}...", flush=True)
    model, tokenizer = load(MODEL_ID)
    sampler = make_sampler(temp=0)

    start = time.perf_counter()

    with (
        output_file.open("a", encoding="utf-8", newline="") as out,
        diagnostic_file.open(
            "a", encoding="utf-8", newline=""
        ) as diag,
    ):
        writer = csv.DictWriter(
            out, fieldnames=OUTPUT_COLUMNS
        )

        diag_writer = csv.DictWriter(
            diag, fieldnames=DIAGNOSTIC_COLUMNS
        )

        if out.tell() == 0:
            writer.writeheader()

        if diag.tell() == 0:
            diag_writer.writeheader()

        for i, row in enumerate(pending, 1):
            prediction, tokens, reason, truncated = (
                generate_translation(
                    model, tokenizer, sampler, row["source"]
                )
            )

            row["prediction"] = prediction

            diagnostic = {
                k: row[k] for k in KEYS
            }

            diagnostic.update({
                "model_id": MODEL_ID,
                "generation_tokens": tokens,
                "finish_reason": reason,
                "potentially_truncated": truncated,
            })

            writer.writerow(row)
            out.flush()

            diag_writer.writerow(diagnostic)
            diag.flush()

            if i % 10 == 0 or i == len(pending):
                elapsed = time.perf_counter() - start

                print(
                    f"{i}/{len(pending)} generated | "
                    f"{elapsed:.1f}s elapsed | "
                    f"{elapsed/i:.2f}s/sentence | "
                    f"last_finish={reason} | "
                    f"truncation_warning={truncated}",
                    flush=True,
                )

    metadata = {
        "model_id": MODEL_ID,
        "source_file": str(SOURCE_FILE),
        "mode": args.mode,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "system_prompt": SYSTEM_PROMPT,
        "output_file": str(output_file),
        "diagnostic_file": str(diagnostic_file),
        "pilot_sampling_seed": (
            42 if args.mode == "pilot" else None
        ),
    }

    metadata_file = OUTPUT_DIR / f"{args.mode}_metadata.json"

    metadata_file.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved: {output_file}")
    print(f"Diagnostics: {diagnostic_file}")


if __name__ == "__main__":
    main()
