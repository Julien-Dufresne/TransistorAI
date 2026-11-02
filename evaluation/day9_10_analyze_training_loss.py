
"""
TransistorAI — Day 10: Training and Validation Loss Analysis

Visualize training and validation losses for:
E1: Technical-only fine-tuning
E2: Mixed-domain fine-tuning

Run:
    python evaluation/day10_analyze_training_loss.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# 1. Define project paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

EXPERIMENTS = {
    "E1 — Technical-only": ROOT / "results/day8-de-en-v2/technical-finetuned/training_history.json",
    "E2 — Mixed-domain": ROOT / "results/day8-de-en-v2/mixed-finetuned/training_history.json",
}

OUTPUT_DIR = ROOT / "results/day9_10_error_analysis/figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load and prepare the histories
# ============================================================

histories = {}

for name, path in EXPERIMENTS.items():

    if not path.exists():
        raise FileNotFoundError(path)

    with open(path, encoding="utf-8") as file:
        records = json.load(file)

    df = pd.DataFrame(records)

    # Hugging Face Trainer stores training and evaluation
    # losses in separate history records.
    train = df.dropna(subset=["loss", "epoch"]).copy()
    validation = df.dropna(
        subset=["eval_loss", "epoch"]
    ).copy()

    if train.empty:
        raise ValueError(f"No training loss found: {name}")

    print(f"\n{name}")
    print(f"Training records: {len(train)}")
    print(f"Validation records: {len(validation)}")

    if not validation.empty:
        print(
            validation[["epoch", "eval_loss"]]
            .to_string(index=False)
        )

    histories[name] = (train, validation)


# ============================================================
# 3. Plot training and validation loss
# ============================================================

fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 5),
    sharey=True,
)

colors = {
    "E1 — Technical-only": "#38BDF8",
    "E2 — Mixed-domain": "#A78BFA",
}

for ax, (name, (train, validation)) in zip(
    axes, histories.items()
):

    # Smooth noisy training losses using a rolling mean.
    train = train.sort_values("epoch")

    train["smoothed_loss"] = (
        train["loss"]
        .rolling(window=100, min_periods=1)
        .mean()
    )

    ax.plot(
        train["epoch"],
        train["smoothed_loss"],
        label="Training loss (smoothed)",
        color=colors[name],
        linewidth=2,
    )

    # Validation loss is measured at evaluation checkpoints.
    if not validation.empty:
        ax.plot(
            validation["epoch"],
            validation["eval_loss"],
            marker="o",
            markersize=7,
            color="#FBBF24",
            linewidth=2,
            label="Validation loss",
        )

    ax.set_title(name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(alpha=0.2)
    ax.legend()

fig.suptitle(
    "TransistorAI — Training vs Validation Loss",
    fontsize=16,
)

fig.tight_layout()

for extension in ("png", "svg"):
    output = OUTPUT_DIR / f"08_training_validation_loss.{extension}"
    fig.savefig(output, dpi=220, bbox_inches="tight")
    print(f"Saved: {output}")

plt.close(fig)


# ============================================================
# 4. Print an interpretation summary
# ============================================================

for name, (train, validation) in histories.items():

    print(f"\n{name}")

    if validation.empty:
        print(
            "No validation loss found. "
            "Cannot assess overfitting from this history."
        )
        continue

    first = validation.sort_values("epoch").iloc[0]["eval_loss"]
    last = validation.sort_values("epoch").iloc[-1]["eval_loss"]

    print(f"First validation loss: {first:.4f}")
    print(f"Final validation loss: {last:.4f}")
    print(f"Change: {last - first:+.4f}")

    if last < first:
        print(
            "Validation loss decreased overall. "
            "No sustained increase between the first "
            "and final checkpoints."
        )
    else:
        print(
            "Validation loss did not improve overall. "
            "Inspect the curve for potential overfitting."
        )

