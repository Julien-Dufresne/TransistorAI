
# ============================================================
# TransistorAI — Day 9 Experiment Visualizations
#
# Compare three translation models:
# E0: Pretrained MarianMT
# E1: Technical-only fine-tuning
# E2: Mixed-domain fine-tuning
#
# Run from project root:
# python -m evaluation.visualize_experiments
#
# Output:
# results/day9_visualizations/
# ============================================================

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Patch


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS = ROOT / "results/day8-de-en-v2"
OUTPUT = ROOT / "results/day9_visualizations"

OUTPUT.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. LOAD SCORES FROM SAVED EXPERIMENT RESULTS
#
# E0 is the pretrained baseline.
# E1 and E2 are the fine-tuned experiments.
#
# Read the actual saved scores rather than manually typing
# values into the visualization code.
# ============================================================

import pandas as pd

SCORE_FILES = {
    "E0": ROOT / "results/day5_scores.csv",
    "E1": RESULTS / "e1_evaluation/scores.csv",
    "E2": RESULTS / "e2_evaluation/scores.csv",
}

DOMAINS = ["technical", "general"]
MODELS = ["E0", "E1", "E2"]

scores = {}

for model, path in SCORE_FILES.items():

    if not path.exists():
        raise FileNotFoundError(
            f"Missing {model} scores: {path}"
        )

    df = pd.read_csv(path)

    # Check that all required columns are present.
    required = {"domain", "bleu", "chrf"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path} is missing columns: {missing}"
        )

    scores[model] = {}

    for domain in DOMAINS:

        rows = df[
            df["domain"].astype(str).str.lower() == domain
        ]

        # Each model must have exactly one score per domain.
        if len(rows) != 1:
            raise ValueError(
                f"{model}: expected one {domain} row, "
                f"found {len(rows)}"
            )

        row = rows.iloc[0]

        scores[model][domain] = {
            "bleu": float(row["bleu"]),
            "chrf": float(row["chrf"]),
        }

print("Loaded E0, E1 and E2 scores successfully.")


# ============================================================
# 3. VISUAL THEME
#
# Dark background, subtle gridlines and consistent colors.
# ============================================================

BG = "#171717"
GRID = "#343434"
TEXT = "#F5F5F5"
MUTED = "#B3B3B3"

COLORS = {
    "E0": "#28549A",   # Blue
    "E1": "#398642",   # Green
    "E2": "#B44F25",   # Burnt orange
}

LABELS = {
    "E0": "E0 Pretrained",
    "E1": "E1 Technical",
    "E2": "E2 Mixed",
}

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "savefig.facecolor": BG,
    "text.color": TEXT,
    "axes.labelcolor": MUTED,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.titlesize": 20,
    "axes.titleweight": "semibold",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "legend.frameon": False,
    "savefig.dpi": 300,
})


# ============================================================
# 4. REUSABLE GRAPH HELPERS
# ============================================================

def make_figure(title, figsize=(12, 6.5)):
    """Create a consistently styled figure."""

    fig, ax = plt.subplots(figsize=figsize)

    ax.set_title(
        title,
        loc="left",
        color=TEXT,
        pad=28,
        fontsize=21,
        fontweight="semibold",
    )

    ax.set_axisbelow(True)
    ax.grid(
        axis="y",
        color=GRID,
        linewidth=0.8,
        linestyle=(0, (4, 4)),
        alpha=0.7,
    )

    ax.tick_params(axis="both", length=0, pad=12)

    return fig, ax


def rounded_bar(ax, x, height, width, color, baseline=0):
    """
    Draw a rounded vertical bar.

    Supports positive and negative heights.
    """

    bottom = min(baseline, baseline + height)
    bar_height = abs(height)

    patch = FancyBboxPatch(
        (x - width / 2, bottom),
        width,
        bar_height,
        boxstyle="round,pad=0,rounding_size=0.12",
        linewidth=0,
        facecolor=color,
        mutation_aspect=1,
        zorder=3,
    )

    ax.add_patch(patch)


def add_legend(ax, models=MODELS):
    """Add a clean legend with model colors."""

    handles = [
        Patch(
            facecolor=COLORS[model],
            edgecolor="none",
            label=LABELS[model],
        )
        for model in models
    ]

    legend = ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(models),
        fontsize=12,
        columnspacing=2.5,
    )

    for text in legend.get_texts():
        text.set_color(TEXT)


def save_graph(fig, filename):
    """Export both high-resolution PNG and vector SVG."""

    fig.tight_layout()

    for extension in ["png", "svg"]:
        path = OUTPUT / f"{filename}.{extension}"

        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            facecolor=BG,
        )

        print(f"Saved: {path}")

    plt.close(fig)


# ============================================================
# 5. GRAPH A / D — GROUPED BLEU AND chrF COMPARISON
# ============================================================

def grouped_metric_chart(metric, title, filename, upper_limit):
    """
    Compare all three models across technical and general
    translation domains.
    """

    fig, ax = make_figure(title)

    centers = np.array([0, 1.9])
    width = 0.46
    offsets = [-0.49, 0, 0.49]

    for model, offset in zip(MODELS, offsets):

        for index, domain in enumerate(DOMAINS):

            value = scores[model][domain][metric]
            x = centers[index] + offset

            rounded_bar(
                ax=ax,
                x=x,
                height=value,
                width=width,
                color=COLORS[model],
            )

            # Show exact values above the bars.
            ax.text(
                x,
                value + 1.0,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                color=TEXT,
                fontsize=11,
                fontweight="medium",
            )

    ax.set_xlim(-0.9, 2.8)
    ax.set_ylim(0, upper_limit)

    ax.set_xticks(
        centers,
        ["Technical", "General"],
    )

    ax.set_yticks(
        np.arange(0, upper_limit + 1, 10)
    )

    ax.set_ylabel(f"{metric.upper()} score")

    add_legend(ax)

    save_graph(fig, filename)


# ============================================================
# 6. GRAPH B — IMPROVEMENT OVER PRETRAINED BASELINE
# ============================================================

def baseline_improvement_chart():
    """
    Display BLEU differences relative to E0.

    Positive values = improvement.
    Negative values = performance degradation.
    """

    fig, ax = make_figure(
        "BLEU Improvement Over Baseline"
    )

    centers = [0, 1.8]
    width = 0.62

    for index, domain in enumerate(DOMAINS):

        baseline = scores["E0"][domain]["bleu"]

        for model, offset in [
            ("E1", -0.34),
            ("E2", 0.34),
        ]:

            delta = (
                scores[model][domain]["bleu"] - baseline
            )

            x = centers[index] + offset

            rounded_bar(
                ax=ax,
                x=x,
                height=delta,
                width=width,
                color=COLORS[model],
            )

            # Position the label above or below the bar.
            label_y = (
                delta + 0.35
                if delta >= 0
                else delta - 0.35
            )

            ax.text(
                x,
                label_y,
                f"{delta:+.2f}",
                ha="center",
                va="bottom" if delta >= 0 else "top",
                color=TEXT,
                fontsize=12,
                fontweight="medium",
            )

    ax.axhline(
        0,
        color="#777777",
        linewidth=1,
        zorder=2,
    )

    ax.set_xlim(-0.9, 2.7)
    ax.set_ylim(-12, 9)

    ax.set_xticks(
        centers,
        ["Technical", "General"],
    )

    ax.set_ylabel("BLEU point change from E0")

    add_legend(ax, ["E1", "E2"])

    save_graph(fig, "bleu_improvement_over_baseline")


# ============================================================
# 7. GRAPH C — TECHNICAL VS GENERAL DOMAIN TRADE-OFF
# ============================================================

def domain_tradeoff_chart():
    """
    Show the relationship between technical and general
    BLEU performance for each model.
    """

    fig, ax = make_figure(
        "Technical Specialization vs Generalization",
        figsize=(10, 7),
    )

    for model in MODELS:

        technical = scores[model]["technical"]["bleu"]
        general = scores[model]["general"]["bleu"]

        ax.scatter(
            general,
            technical,
            s=260,
            color=COLORS[model],
            edgecolors=BG,
            linewidths=2,
            zorder=4,
        )

        ax.annotate(
            LABELS[model],
            (general, technical),
            xytext=(12, 10),
            textcoords="offset points",
            color=TEXT,
            fontsize=12,
            fontweight="medium",
        )

    ax.set_xlabel("General-domain BLEU")
    ax.set_ylabel("Technical-domain BLEU")

    ax.set_xlim(27, 46)
    ax.set_ylim(32, 44)

    ax.grid(
        axis="both",
        color=GRID,
        linestyle=(0, (4, 4)),
        alpha=0.7,
    )

    # Explain which direction indicates better performance.
    ax.text(
        0.98,
        0.04,
        "Higher and further right = better",
        transform=ax.transAxes,
        ha="right",
        color=MUTED,
        fontsize=10,
    )

    save_graph(fig, "domain_tradeoff")


# ============================================================
# 8. GENERATE ALL VISUALIZATIONS
# ============================================================

def main():

    grouped_metric_chart(
        metric="bleu",
        title="BLEU Performance by Domain",
        filename="bleu_performance_by_domain",
        upper_limit=50,
    )

    baseline_improvement_chart()

    grouped_metric_chart(
        metric="chrf",
        title="chrF Comparison by Domain",
        filename="chrf_comparison_by_domain",
        upper_limit=80,
    )

    domain_tradeoff_chart()

    print("\nAll four TransistorAI graphs generated!")
    print(f"Output directory: {OUTPUT}")


if __name__ == "__main__":
    main()
