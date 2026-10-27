
# Day 7 — Compare pretrained and fine-tuned models.
# Export the comparison as CSV and Markdown.

import json
from pathlib import Path

import pandas as pd
from pathlib import Path
from pandas.errors import EmptyDataError


def main():
    """Compare Day 5 baseline and Day 7 fine-tuned evaluation scores."""

    # STEP 1 — Load the evaluation configuration.
    with open("configs/day7_eval_config.json") as file:
        config = json.load(file)

    # STEP 2 — Load Day 5 pretrained baseline scores.
    baseline = pd.read_csv(
        config["baseline_scores_path"]
    )

    
    # STEP 3 — Load Day 7 fine-tuned scores.


    scores_path = Path(config["output"]["scores"])

    # Check whether the file exists.
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Day 7 scores file not found: {scores_path}"
        )

    # Check whether the file is empty.
    if scores_path.stat().st_size == 0:
        raise ValueError(
            f"Day 7 scores file is empty: {scores_path}. "
            "Run the Day 7 evaluation before comparing models."
        )

    # Load the scores.
    try:
        finetuned = pd.read_csv(scores_path)
    except EmptyDataError as exc:
        raise ValueError(
            f"Day 7 scores file has no readable columns: {scores_path}"
        ) from exc

    # STEP 4 — Define columns identifying each experiment.
    # Dataset paths must match between Day 5 and Day 7.
    keys = ["language_pair", "domain", "dataset"]

    # Define the columns required for comparison.
    required = set(keys + ["bleu", "chrf"])

    # Verify that both datasets contain the expected columns.
    for name, df in [
        ("baseline", baseline),
        ("fine_tuned", finetuned),
    ]:
        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing columns in {name}: {sorted(missing)}"
            )

        # Prevent duplicate experiment comparisons.
        if df.duplicated(subset=keys).any():
            raise ValueError(
                f"Duplicate experiment keys in {name}"
            )

    # STEP 5 — Match fine-tuned results with baseline results.
    comparison = finetuned.merge(
        baseline[keys + ["bleu", "chrf"]],
        on=keys,
        how="left",
        suffixes=("_finetuned", "_baseline"),
        validate="one_to_one",
    )

    # Stop if a matching baseline result is missing.
    if comparison[
        ["bleu_baseline", "chrf_baseline"]
    ].isna().any().any():
        raise ValueError(
            "Missing matching Day 5 baseline results."
        )

    # STEP 6 — Calculate absolute score differences.
    # Positive values indicate improvement after fine-tuning.
    # Negative values indicate performance degradation.
    comparison["bleu_delta"] = (
        comparison["bleu_finetuned"]
        - comparison["bleu_baseline"]
    )

    comparison["chrf_delta"] = (
        comparison["chrf_finetuned"]
        - comparison["chrf_baseline"]
    )

    # STEP 7 — Save the comparison as CSV.
    comparison_path = Path(config["output"]["comparison"])

    # Create the output directory if necessary.
    comparison_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    # STEP 8 — Generate a Markdown comparison report.
    report_path = Path(
        "results/day7_comparison_report.md"
    )

    # Define the columns displayed in the report.
    columns = [
        "language_pair",
        "domain",
        "bleu_baseline",
        "bleu_finetuned",
        "bleu_delta",
        "chrf_baseline",
        "chrf_finetuned",
        "chrf_delta",
    ]

    # Build the Markdown report.
    report_lines = [
        "# Day 7 — Fine-Tuned Model Evaluation",
        "",
        "## Day 5 vs Day 7 Comparison",
        "",
        "This report compares the pretrained MarianMT baseline "
        "with the fine-tuned model.",
        "",
        "Positive deltas indicate improvement; "
        "negative deltas indicate degradation.",
        "",
        "| Language | Domain | BLEU Baseline | BLEU Fine-Tuned "
        "| BLEU Delta | chrF Baseline | chrF Fine-Tuned | chrF Delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    # Add one row per evaluated language pair and domain.
    for _, row in comparison[columns].iterrows():
        report_lines.append(
            f"| {row['language_pair']} "
            f"| {row['domain']} "
            f"| {row['bleu_baseline']:.4f} "
            f"| {row['bleu_finetuned']:.4f} "
            f"| {row['bleu_delta']:+.4f} "
            f"| {row['chrf_baseline']:.4f} "
            f"| {row['chrf_finetuned']:.4f} "
            f"| {row['chrf_delta']:+.4f} |"
        )

    # STEP 9 — Add an interpretation of the results.
    report_lines.extend([
        "",
        "## Interpretation",
        "",
    ])

    for _, row in comparison.iterrows():

        # Determine whether BLEU improved or declined.
        bleu_status = (
            "improved" if row["bleu_delta"] > 0
            else "declined" if row["bleu_delta"] < 0
            else "remained unchanged"
        )

        # Determine whether chrF improved or declined.
        chrf_status = (
            "improved" if row["chrf_delta"] > 0
            else "declined" if row["chrf_delta"] < 0
            else "remained unchanged"
        )

        report_lines.append(
            f"### {row['language_pair']} — {row['domain']}\n\n"
            f"- BLEU {bleu_status} by "
            f"{abs(row['bleu_delta']):.4f} points.\n"
            f"- chrF {chrf_status} by "
            f"{abs(row['chrf_delta']):.4f} points.\n"
        )

    # STEP 10 — Save the Markdown report.
    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    report_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    # STEP 11 — Display only confirmation messages.
    print(f"CSV comparison saved: {comparison_path}")
    print(f"Markdown report saved: {report_path}")


if __name__ == "__main__":
    main()
