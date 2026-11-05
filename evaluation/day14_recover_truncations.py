import csv
import json
from pathlib import Path

import pandas as pd
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from day14_generate_llm import MODEL_ID, SYSTEM_PROMPT, KEYS

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results/day14_llm_baseline"
OUTPUT = BASE / "truncation_recovery_384.csv"
MAX_TOKENS = 384

FIELDS = KEYS + [
    "source",
    "reference",
    "original_prediction",
    "recovered_prediction",
    "original_generation_tokens",
    "recovery_generation_tokens",
    "original_finish_reason",
    "recovery_finish_reason",
    "recovery_potentially_truncated",
]


def main():
    predictions = pd.read_csv(
        BASE / "predictions.csv",
        keep_default_na=False,
    )

    diagnostics = pd.read_csv(
        BASE / "generation_diagnostics.csv",
        keep_default_na=False,
    )

    if predictions.duplicated(KEYS).any():
        raise ValueError("Duplicate prediction keys")

    if diagnostics.duplicated(KEYS).any():
        raise ValueError("Duplicate diagnostic keys")

    flagged = diagnostics[
        diagnostics["potentially_truncated"]
        .astype(str)
        .str.lower()
        .eq("true")
    ].copy()

    review = flagged.merge(
        predictions,
        on=KEYS,
        validate="one_to_one",
        suffixes=("_diagnostic", ""),
    )

    if len(review) != 11:
        raise ValueError(
            f"Expected 11 flagged examples, found {len(review)}"
        )

    completed = set()

    if OUTPUT.exists():
        existing = pd.read_csv(OUTPUT, keep_default_na=False)

        if existing.duplicated(KEYS).any():
            raise ValueError("Duplicate recovery keys")

        completed = set(
            existing[KEYS].itertuples(index=False, name=None)
        )

    pending = [
        row
        for row in review.to_dict("records")
        if tuple(row[key] for key in KEYS) not in completed
    ]

    print(f"Flagged examples: {len(review)}")
    print(f"Previously recovered: {len(completed)}")
    print(f"Pending: {len(pending)}", flush=True)

    if not pending:
        return

    model, tokenizer = load(MODEL_ID)
    sampler = make_sampler(temp=0)

    with OUTPUT.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDS)

        if file.tell() == 0:
            writer.writeheader()

        for i, row in enumerate(pending, 1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["source"]},
            ]

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            chunks = []
            last = None

            for response in stream_generate(
                model,
                tokenizer,
                prompt=prompt,
                max_tokens=MAX_TOKENS,
                sampler=sampler,
            ):
                chunks.append(response.text)
                last = response

            if last is None:
                raise RuntimeError("No generation response")

            recovered = "".join(chunks).strip()

            if not recovered:
                raise ValueError(
                    f"Empty recovery for {row['sentence_id']}"
                )

            record = {
                **{key: row[key] for key in KEYS},
                "source": row["source"],
                "reference": row["reference"],
                "original_prediction": row["prediction"],
                "recovered_prediction": recovered,
                "original_generation_tokens": row["generation_tokens"],
                "recovery_generation_tokens": last.generation_tokens,
                "original_finish_reason": row["finish_reason"],
                "recovery_finish_reason": last.finish_reason,
                "recovery_potentially_truncated": (
                    last.finish_reason in {"length", "max_tokens"}
                    or last.generation_tokens >= MAX_TOKENS
                ),
            }

            writer.writerow(record)
            file.flush()

            print(
                f"{i}/{len(pending)} | "
                f"ID={row['sentence_id']} | "
                f"tokens={last.generation_tokens} | "
                f"finish={last.finish_reason}",
                flush=True,
            )

    metadata = {
        "model_id": MODEL_ID,
        "max_tokens": MAX_TOKENS,
        "temperature": 0,
        "system_prompt": SYSTEM_PROMPT,
        "original_max_tokens": 128,
        "output_file": str(OUTPUT),
    }

    (BASE / "truncation_recovery_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"\nRecovery saved to {OUTPUT}")


if __name__ == "__main__":
    main()
