
# Day 7 — Generate the reproducible evaluation report.

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import sacrebleu


def main():

    # STEP 1 — Load evaluation configuration.
    with open("configs/day7_eval_config.json") as file:
        config = json.load(file)

    # STEP 2 — Load model comparison results.
    df = pd.read_csv(
        config["output"]["comparison"]
    )

    # STEP 3 — Initialize the Markdown report.
    lines = [
        "# Day 7 — Fine-Tuned Model Evaluation",
        "",
        "## Objective",
        "",
        "Evaluate fine-tuning gains on technical translation",
        "and measure general-domain performance changes.",
        "",
        "## Experiment Configuration",
        "",
        f"- Language pair: {config['language_pair']}",
        f"- Model: {config['model_path']}",
        f"- Seed: {config['seed']}",
        f"- SacreBLEU version: {sacrebleu.__version__}",
        f"- Decoding: `{json.dumps(config['decoding'])}`",
        f"- Generated (UTC): {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Model Comparison",
        "",
        "| Domain | BLEU baseline | BLEU fine-tuned | "
        "BLEU delta | chrF baseline | chrF fine-tuned | chrF delta |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    # STEP 4 — Write one comparison row per domain.
    for _, row in df.iterrows():

        lines.append(
            f"| {row['domain']} "
            f"| {row['bleu_baseline']:.4f} "
            f"| {row['bleu_finetuned']:.4f} "
            f"| {row['bleu_delta']:+.4f} "
            f"| {row['chrf_baseline']:.4f} "
            f"| {row['chrf_finetuned']:.4f} "
            f"| {row['chrf_delta']:+.4f} |"
        )

    # STEP 5 — Add interpretation of the results.
    lines.extend([
        "",
        "## Interpretation",
        "",
    ])

    for _, row in df.iterrows():

        bleu_delta = row["bleu_delta"]
        chrf_delta = row["chrf_delta"]

        # Describe metric changes without assuming causality.
        bleu_status = (
            "improved" if bleu_delta > 0
            else "declined" if bleu_delta < 0
            else "remained unchanged"
        )

        chrf_status = (
            "improved" if chrf_delta > 0
            else "declined" if chrf_delta < 0
            else "remained unchanged"
        )

        lines.append(
            f"- **{row['domain'].title()}**: "
            f"BLEU {bleu_status} ({bleu_delta:+.4f}); "
            f"chrF {chrf_status} ({chrf_delta:+.4f})."
        )

    # STEP 6 — Add sections for qualitative findings.
    lines.extend([
        "",
        "## Qualitative Error Analysis",
        "",
        "Add findings from day7_manual_inspection.csv.",
        "",
        "## Limitations",
        "",
        "- Automatic metrics do not fully measure adequacy.",
        "- Small score changes may not be statistically significant.",
        "- Dataset and decoding settings must match Day 5.",
        "",
        "## Conclusion",
        "",
        "Assess technical gains and general-domain trade-offs",
        "after reviewing automatic and manual evaluation.",
        "",
    ])

    # STEP 7 — Save the report as Markdown.
    output_path = Path(
        config["output"]["report"]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
