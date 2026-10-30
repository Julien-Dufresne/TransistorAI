
"""
TransistorAI — Day 9–10 Qualitative Error Analysis

Compare E1 (technical-only fine-tuning) and E2 (mixed-domain
fine-tuning) using 50 manually annotated translations.

Outputs:
    - model_preferences.csv
    - severity_distribution.csv
    - error_category_distribution.csv
    - meaning_preservation.csv
    - qualitative_summary.md
    - figures/model_preferences.png
    - figures/error_categories.png
    - figures/error_severity.png

The 50 examples were deliberately enriched for disagreements.
Results are descriptive, not representative population estimates.
"""

from pathlib import Path
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. Define project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    PROJECT_ROOT / "results" / "day9_10_error_analysis"
)

INPUT_PATH = OUTPUT_DIR / "manual_review_annotated.csv"

FIGURE_DIR = OUTPUT_DIR / "figures"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. Configure visual appearance
# ============================================================

BACKGROUND = "#111827"
TEXT_COLOR = "#F9FAFB"
GRID_COLOR = "#374151"

E1_COLOR = "#38BDF8"
E2_COLOR = "#A78BFA"
TIE_COLOR = "#9CA3AF"

plt.rcParams.update({
    "figure.facecolor": BACKGROUND,
    "axes.facecolor": BACKGROUND,
    "savefig.facecolor": BACKGROUND,
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "font.size": 11,
})


def save_figure(filename):
    """Save a figure as both PNG and editable SVG."""
    for extension in ("png", "svg"):
        plt.savefig(
            FIGURE_DIR / f"{filename}.{extension}",
            dpi=200,
            bbox_inches="tight",
        )
    plt.close()


# ============================================================
# 3. Load and validate annotations
# ============================================================

# Keep the literal "None" instead of converting it to NaN.
df = pd.read_csv(INPUT_PATH, keep_default_na=False)

required_columns = [
    "review_id",
    "domain",
    "e1_error_category",
    "e2_error_category",
    "e1_severity",
    "e2_severity",
    "e1_meaning_preserved",
    "e2_meaning_preserved",
    "preferred_model",
]

missing_columns = set(required_columns) - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

if len(df) != 50:
    raise ValueError(
        f"Expected 50 annotated rows, found {len(df)}"
    )

if df["review_id"].duplicated().any():
    raise ValueError("Duplicate review IDs detected.")

allowed_severity = {
    "None", "Minor", "Moderate", "Major"
}

allowed_meaning = {
    "Yes", "Partially", "No"
}

allowed_preferences = {
    "E1", "E2", "Tie"
}

allowed_domains = {
    "technical", "general"
}

for model in ("e1", "e2"):
    severity_column = f"{model}_severity"
    meaning_column = f"{model}_meaning_preserved"
    category_column = f"{model}_error_category"

    invalid_severity = (
        set(df[severity_column]) - allowed_severity
    )

    invalid_meaning = (
        set(df[meaning_column]) - allowed_meaning
    )

    if invalid_severity:
        raise ValueError(
            f"{severity_column}: invalid values "
            f"{invalid_severity}"
        )

    if invalid_meaning:
        raise ValueError(
            f"{meaning_column}: invalid values "
            f"{invalid_meaning}"
        )

    # Each row must have either no error or a classified error.
    no_error = df[severity_column].eq("None")
    no_category = df[category_column].eq("None")

    inconsistent = no_error != no_category

    if inconsistent.any():
        raise ValueError(
            f"Inconsistent error annotations for {model}: "
            f"review IDs "
            f"{df.loc[inconsistent, 'review_id'].tolist()}"
        )

if not set(df["preferred_model"]).issubset(
    allowed_preferences
):
    raise ValueError("Invalid preferred_model value.")

if not set(df["domain"]).issubset(allowed_domains):
    raise ValueError("Unexpected domain value.")

print(f"Loaded {len(df)} annotated translations.")
print("Annotation validation passed.")


# ============================================================
# 4. Calculate model preferences
# ============================================================

preference_counts = pd.crosstab(
    df["domain"],
    df["preferred_model"],
)

preference_counts = preference_counts.reindex(
    index=["technical", "general"],
    columns=["E1", "E2", "Tie"],
    fill_value=0,
)

preference_counts.loc["Total"] = (
    preference_counts.sum(axis=0)
)

preference_counts.to_csv(
    OUTPUT_DIR / "model_preferences.csv"
)

print("\nMODEL PREFERENCES")
print(preference_counts)


# ============================================================
# 5. Calculate error severity distributions
# ============================================================

severity_order = [
    "None", "Minor", "Moderate", "Major"
]

severity_records = []

for domain in ("technical", "general"):
    subset = df[df["domain"] == domain]

    for model in ("e1", "e2"):
        counts = (
            subset[f"{model}_severity"]
            .value_counts()
            .reindex(severity_order, fill_value=0)
        )

        for severity, count in counts.items():
            severity_records.append({
                "domain": domain,
                "model": model.upper(),
                "severity": severity,
                "count": int(count),
            })

severity_df = pd.DataFrame(severity_records)

severity_df.to_csv(
    OUTPUT_DIR / "severity_distribution.csv",
    index=False,
)

print("\nERROR SEVERITY")
print(severity_df.to_string(index=False))


# ============================================================
# 6. Calculate multi-label error categories
# ============================================================

# A translation can contain multiple errors, for example:
# "Mistranslation; Grammar"
#
# Count each category separately.
# Consequently, category totals can exceed sentence counts.

category_records = []

for domain in ("technical", "general"):
    subset = df[df["domain"] == domain]

    for model in ("e1", "e2"):
        counter = Counter()

        for value in subset[f"{model}_error_category"]:
            if value == "None":
                continue

            categories = [
                item.strip()
                for item in value.split(";")
                if item.strip()
            ]

            # Count a category once per translation.
            counter.update(set(categories))

        for category, count in sorted(counter.items()):
            category_records.append({
                "domain": domain,
                "model": model.upper(),
                "error_category": category,
                "count": count,
            })

category_df = pd.DataFrame(category_records)

category_df.to_csv(
    OUTPUT_DIR / "error_category_distribution.csv",
    index=False,
)

print("\nERROR CATEGORIES")
print(category_df.to_string(index=False))


# ============================================================
# 7. Calculate meaning preservation
# ============================================================

meaning_records = []

for domain in ("technical", "general"):
    subset = df[df["domain"] == domain]

    for model in ("e1", "e2"):
        counts = subset[
            f"{model}_meaning_preserved"
        ].value_counts()

        for meaning in ("Yes", "Partially", "No"):
            meaning_records.append({
                "domain": domain,
                "model": model.upper(),
                "meaning_preserved": meaning,
                "count": int(counts.get(meaning, 0)),
            })

meaning_df = pd.DataFrame(meaning_records)

meaning_df.to_csv(
    OUTPUT_DIR / "meaning_preservation.csv",
    index=False,
)


# ============================================================
# 8. Graph 1 — Model preferences by domain
# ============================================================

fig, ax = plt.subplots(figsize=(10, 6))

plot_data = preference_counts.loc[
    ["technical", "general"]
]

x = np.arange(len(plot_data))
width = 0.24

for offset, model, color in [
    (-width, "E1", E1_COLOR),
    (0, "E2", E2_COLOR),
    (width, "Tie", TIE_COLOR),
]:
    bars = ax.bar(
        x + offset,
        plot_data[model],
        width,
        label=model,
        color=color,
    )

    ax.bar_label(bars, padding=4)

ax.set_xticks(x)
ax.set_xticklabels(["Technical", "General"])
ax.set_ylabel("Number of reviewed translations")
ax.set_title(
    "Human Preference — E1 vs E2",
    fontsize=16,
    pad=20,
)
ax.legend(frameon=False)
ax.set_ylim(0, plot_data.to_numpy().max() + 5)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

save_figure("model_preferences")


# ============================================================
# 9. Graph 2 — Error category frequencies
# ============================================================

category_plot = (
    category_df.groupby(
        ["error_category", "model"]
    )["count"]
    .sum()
    .unstack(fill_value=0)
)

category_plot = category_plot.reindex(
    columns=["E1", "E2"],
    fill_value=0,
)

category_plot = category_plot.loc[
    category_plot.sum(axis=1).sort_values().index
]

fig, ax = plt.subplots(
    figsize=(11, max(5, len(category_plot) * 0.65))
)

y = np.arange(len(category_plot))
height = 0.36

ax.barh(
    y - height / 2,
    category_plot["E1"],
    height,
    label="E1",
    color=E1_COLOR,
)

ax.barh(
    y + height / 2,
    category_plot["E2"],
    height,
    label="E2",
    color=E2_COLOR,
)

ax.set_yticks(y)
ax.set_yticklabels(category_plot.index)
ax.set_xlabel("Number of annotated occurrences")
ax.set_title(
    "Error Categories — E1 vs E2",
    fontsize=16,
    pad=20,
)
ax.legend(frameon=False)
ax.grid(axis="x", alpha=0.2)
ax.set_axisbelow(True)

save_figure("error_categories")


# ============================================================
# 10. Graph 3 — Severity distribution
# ============================================================

severity_plot = (
    severity_df.groupby(
        ["severity", "model"]
    )["count"]
    .sum()
    .unstack(fill_value=0)
    .reindex(severity_order)
    .reindex(columns=["E1", "E2"])
)

fig, ax = plt.subplots(figsize=(10, 6))

x = np.arange(len(severity_order))
width = 0.35

bars_e1 = ax.bar(
    x - width / 2,
    severity_plot["E1"],
    width,
    label="E1",
    color=E1_COLOR,
)

bars_e2 = ax.bar(
    x + width / 2,
    severity_plot["E2"],
    width,
    label="E2",
    color=E2_COLOR,
)

ax.bar_label(bars_e1, padding=3)
ax.bar_label(bars_e2, padding=3)

ax.set_xticks(x)
ax.set_xticklabels(severity_order)
ax.set_ylabel("Number of reviewed translations")
ax.set_title(
    "Translation Error Severity",
    fontsize=16,
    pad=20,
)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
ax.set_axisbelow(True)

save_figure("error_severity")


# ============================================================
# 11. Generate Markdown summary
# ============================================================

total_preferences = preference_counts.loc["Total"]

report = f"""# Day 9–10 — Qualitative Error Analysis

## Dataset

- Reviewed translations: {len(df)}
- Technical: {(df["domain"] == "technical").sum()}
- General: {(df["domain"] == "general").sum()}
- Models: E1 (technical-only), E2 (mixed-domain)

## Model preferences

- E1 preferred: {total_preferences["E1"]}
- E2 preferred: {total_preferences["E2"]}
- Tie: {total_preferences["Tie"]}

## Severity distribution

{severity_plot.to_markdown()}

## Methodology

The analysis uses a manually reviewed sample enriched for
disagreements between E1 and E2.

Error categories are multi-label. Each category is counted
at most once per translation.

Model preferences and error frequencies describe this
selected sample and must not be generalized to the full
test population without representative sampling.

## Generated artifacts

- model_preferences.csv
- severity_distribution.csv
- error_category_distribution.csv
- meaning_preservation.csv
- figures/model_preferences.png
- figures/error_categories.png
- figures/error_severity.png
"""

report_path = OUTPUT_DIR / "qualitative_summary.md"

report_path.write_text(
    report,
    encoding="utf-8",
)

print("\nAnalysis completed successfully.")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Summary report: {report_path}")
