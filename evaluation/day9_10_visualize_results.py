
"""
TransistorAI — Day 9–10 Final Visualizations

Generates seven publication-ready graphs comparing:
    E0: Pretrained MarianMT
    E1: Technical-only fine-tuning
    E2: Mixed-domain fine-tuning

Quantitative data: E0/E1/E2 evaluation scores.
Qualitative data: 50 manually annotated translations.

Run from project root:
    python evaluation/day9_10_visualize_results.py
"""

from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. Define paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = ROOT / "results" / "day9_10_error_analysis"

ANNOTATIONS_PATH = (
    RESULTS_DIR / "manual_review_annotated.csv"
)

FIGURES_DIR = RESULTS_DIR / "figures"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Load experiment scores
# ============================================================

# These are the previously recorded E0/E1/E2 results.
# Keep the values explicit for a reproducible report.
# Update them only if you rerun your benchmark.

scores = pd.DataFrame([
    {
        "model": "E0",
        "domain": "Technical",
        "BLEU": 35.1068,
        "chrF": 57.9039,
    },
    {
        "model": "E0",
        "domain": "General",
        "BLEU": 40.4074,
        "chrF": 65.4856,
    },
    {
        "model": "E1",
        "domain": "Technical",
        "BLEU": 40.5126,
        "chrF": 61.8439,
    },
    {
        "model": "E1",
        "domain": "General",
        "BLEU": 30.6843,
        "chrF": 57.3025,
    },
    {
        "model": "E2",
        "domain": "Technical",
        "BLEU": 39.8935,
        "chrF": 61.4829,
    },
    {
        "model": "E2",
        "domain": "General",
        "BLEU": 35.6534,
        "chrF": 61.6582,
    },
])

# Load annotations while preserving literal "None".
reviews = pd.read_csv(
    ANNOTATIONS_PATH,
    keep_default_na=False,
)

# Confirm the qualitative dataset is complete.
assert len(reviews) == 50, (
    f"Expected 50 reviews, got {len(reviews)}"
)


# ============================================================
# 3. Configure the dark visual theme
# ============================================================

BG = "#0B1220"
PANEL = "#111C2E"
WHITE = "#F8FAFC"
MUTED = "#94A3B8"
GRID = "#334155"

COLORS = {
    "E0": "#94A3B8",
    "E1": "#38BDF8",
    "E2": "#A78BFA",
    "Tie": "#64748B",
}

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": PANEL,
    "savefig.facecolor": BG,
    "text.color": WHITE,
    "axes.labelcolor": WHITE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.edgecolor": GRID,
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
})


def style_axis(ax, axis="y"):
    """Apply consistent grid and axis styling."""

    ax.grid(
        axis=axis,
        color=GRID,
        alpha=0.35,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_visible(False)


def save_chart(fig, name):
    """Export each chart to PNG and SVG."""

    for extension in ("png", "svg"):
        path = FIGURES_DIR / f"{name}.{extension}"

        fig.savefig(
            path,
            dpi=220,
            bbox_inches="tight",
            facecolor=BG,
        )

        print(f"Saved: {path.relative_to(ROOT)}")

    plt.close(fig)


# ============================================================
# 4. Graphs 01 and 02 — BLEU and chrF
# ============================================================

def plot_metric(metric, filename):

    domains = ["Technical", "General"]
    models = ["E0", "E1", "E2"]

    fig, ax = plt.subplots(figsize=(11, 6.5))

    x = np.arange(len(domains))
    width = 0.23

    for i, model in enumerate(models):

        values = [
            scores.loc[
                (scores["model"] == model)
                & (scores["domain"] == domain),
                metric,
            ].iloc[0]
            for domain in domains
        ]

        bars = ax.bar(
            x + (i - 1) * width,
            values,
            width,
            color=COLORS[model],
            label=model,
        )

        ax.bar_label(
            bars,
            fmt="%.2f",
            padding=5,
            color=WHITE,
            fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(domains)

    ax.set_ylabel(f"{metric} score")

    ax.set_title(
        f"{metric} Performance by Domain",
        pad=22,
    )

    ax.set_ylim(
        0,
        max(scores[metric]) * 1.17,
    )

    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper right",
    )

    style_axis(ax)

    fig.tight_layout()

    save_chart(fig, filename)


plot_metric("BLEU", "01_bleu_by_domain")
plot_metric("chrF", "02_chrf_by_domain")


# ============================================================
# 5. Graph 03 — BLEU changes from E0
# ============================================================

baseline = (
    scores[scores["model"] == "E0"]
    .set_index("domain")["BLEU"]
)

fig, ax = plt.subplots(figsize=(11, 6.5))

domains = ["Technical", "General"]
x = np.arange(len(domains))
width = 0.32

for offset, model in [
    (-width / 2, "E1"),
    (width / 2, "E2"),
]:

    values = [
        scores.loc[
            (scores["model"] == model)
            & (scores["domain"] == domain),
            "BLEU",
        ].iloc[0] - baseline[domain]
        for domain in domains
    ]

    bars = ax.bar(
        x + offset,
        values,
        width,
        label=model,
        color=COLORS[model],
    )

    for bar, value in zip(bars, values):

        ax.annotate(
            f"{value:+.2f}",
            (
                bar.get_x() + bar.get_width() / 2,
                value,
            ),
            xytext=(0, 7 if value >= 0 else -17),
            textcoords="offset points",
            ha="center",
            color=WHITE,
        )

ax.axhline(0, color=MUTED, linewidth=1)

ax.set_xticks(x)
ax.set_xticklabels(domains)
ax.set_ylabel("BLEU points vs E0")

ax.set_title(
    "Fine-Tuning Impact Relative to Baseline",
    pad=22,
)

ax.legend(frameon=False)
style_axis(ax)

fig.tight_layout()

save_chart(fig, "03_bleu_delta_from_baseline")


# ============================================================
# 6. Graph 04 — Domain trade-off
# ============================================================

fig, ax = plt.subplots(figsize=(10, 7))

for model in ["E0", "E1", "E2"]:

    model_scores = (
        scores[scores["model"] == model]
        .set_index("domain")
    )

    technical = model_scores.loc["Technical", "BLEU"]
    general = model_scores.loc["General", "BLEU"]

    ax.scatter(
        general,
        technical,
        s=240,
        color=COLORS[model],
        edgecolor=WHITE,
        linewidth=1,
        zorder=3,
    )

    ax.annotate(
        model,
        (general, technical),
        xytext=(9, 9),
        textcoords="offset points",
        fontsize=13,
        fontweight="bold",
    )

ax.set_xlabel("General-domain BLEU")
ax.set_ylabel("Technical-domain BLEU")

ax.set_title(
    "Technical Specialization vs Generalization",
    pad=22,
)

style_axis(ax, axis="both")

fig.tight_layout()

save_chart(fig, "04_domain_tradeoff")


# ============================================================
# 7. Graph 05 — Human model preferences
# ============================================================

preferences = pd.crosstab(
    reviews["domain"],
    reviews["preferred_model"],
).reindex(
    index=["technical", "general"],
    columns=["E1", "E2", "Tie"],
    fill_value=0,
)

fig, ax = plt.subplots(figsize=(11, 6.5))

x = np.arange(2)
width = 0.24

for i, model in enumerate(["E1", "E2", "Tie"]):

    bars = ax.bar(
        x + (i - 1) * width,
        preferences[model],
        width,
        label=model,
        color=COLORS[model],
    )

    ax.bar_label(
        bars,
        padding=4,
        color=WHITE,
    )

ax.set_xticks(x)
ax.set_xticklabels(["Technical", "General"])

ax.set_ylabel("Number of reviewed translations")

ax.set_title(
    "Manual Evaluation — Preferred Model",
    pad=22,
)

ax.legend(frameon=False)
style_axis(ax)

fig.text(
    0.5,
    0.01,
    "50 selected examples; disagreement-enriched sample.",
    ha="center",
    color=MUTED,
    fontsize=9,
)

fig.tight_layout(rect=[0, 0.04, 1, 1])

save_chart(fig, "05_model_preferences")


# ============================================================
# 8. Graph 06 — Error severity
# ============================================================

severity_levels = [
    "None", "Minor", "Moderate", "Major"
]

fig, ax = plt.subplots(figsize=(11, 6.5))

x = np.arange(len(severity_levels))
width = 0.35

for offset, model in [
    (-width / 2, "E1"),
    (width / 2, "E2"),
]:

    counts = (
        reviews[f"{model.lower()}_severity"]
        .value_counts()
        .reindex(severity_levels, fill_value=0)
    )

    bars = ax.bar(
        x + offset,
        counts.values,
        width,
        label=model,
        color=COLORS[model],
    )

    ax.bar_label(
        bars,
        padding=4,
        color=WHITE,
    )

ax.set_xticks(x)
ax.set_xticklabels(severity_levels)

ax.set_ylabel("Number of reviewed translations")

ax.set_title(
    "Error Severity — E1 vs E2",
    pad=22,
)

ax.legend(frameon=False)
style_axis(ax)

fig.text(
    0.5,
    0.01,
    "Manual annotations on 50 selected examples.",
    ha="center",
    color=MUTED,
    fontsize=9,
)

fig.tight_layout(rect=[0, 0.04, 1, 1])

save_chart(fig, "06_error_severity")


# ============================================================
# 9. Graph 07 — Error category frequencies
# ============================================================

# Error categories may contain multiple labels separated
# by semicolons, e.g. "Terminology; Mistranslation".
# Count each category once per translation.

category_counts = {}

for model in ["E1", "E2"]:

    counter = Counter()

    column = f"{model.lower()}_error_category"

    for value in reviews[column]:

        if value == "None":
            continue

        labels = {
            label.strip()
            for label in value.split(";")
            if label.strip()
        }

        counter.update(labels)

    category_counts[model] = counter

categories = sorted(
    set(category_counts["E1"])
    | set(category_counts["E2"])
)

y = np.arange(len(categories))
height = 0.36

fig, ax = plt.subplots(
    figsize=(11, max(6, len(categories) * 0.75))
)

for offset, model in [
    (-height / 2, "E1"),
    (height / 2, "E2"),
]:

    values = [
        category_counts[model].get(category, 0)
        for category in categories
    ]

    bars = ax.barh(
        y + offset,
        values,
        height,
        label=model,
        color=COLORS[model],
    )

    ax.bar_label(
        bars,
        padding=4,
        color=WHITE,
    )

ax.set_yticks(y)
ax.set_yticklabels(categories)

ax.set_xlabel("Annotated category occurrences")

ax.set_title(
    "Translation Error Categories",
    pad=22,
)

ax.legend(frameon=False)
style_axis(ax, axis="x")

fig.text(
    0.5,
    0.01,
    "Multi-label categories; one translation may contribute to multiple categories.",
    ha="center",
    color=MUTED,
    fontsize=9,
)

fig.tight_layout(rect=[0, 0.04, 1, 1])

save_chart(fig, "07_error_categories")


# ============================================================
# 10. Completion summary
# ============================================================

print("\nDay 9–10 visualizations completed.")

print(
    f"Figures saved in: "
    f"{FIGURES_DIR.relative_to(ROOT)}"
)

print("Generated 7 PNG and 7 SVG files.")
