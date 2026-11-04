
"""
Day 12 — Terminology and placeholder evaluation.

Compare E0, E1 and E2 on identical German-English test examples.

Outputs:
- terminology_results.csv
- flagged_examples.csv
- summary_metrics.csv
"""


# ============================================================
# DAY 12 — IMPORTS AND PROJECT CONFIGURATION
# ============================================================

# Import standard Python libraries.
from pathlib import Path
import sys

# Import pandas for processing evaluation results.
import pandas as pd

# Resolve the project root using this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Make the project root available for package imports.
# This allows the script to run directly or as a module.
sys.path.insert(0, str(PROJECT_ROOT))

# Import all Day 12 evaluation functions together.
from evaluation.day12_terminology_checks import (
    check_terminology,
    check_placeholders,
    check_numbers,
    check_units,
    check_protected_content,  # Day 12: URLs and file paths
)



# Define the prediction files for all three models.
MODEL_FILES = {
    "E0": PROJECT_ROOT
    / "results/day11_evaluation/e0_generation/predictions.csv",

    "E1": PROJECT_ROOT
    / "results/day8-de-en-v2/e1_evaluation/predictions.csv",

    "E2": PROJECT_ROOT
    / "results/day8-de-en-v2/e2_evaluation/predictions.csv",
}

# Define the approved terminology glossary.
GLOSSARY_PATH = (
    PROJECT_ROOT
    / "data/terminology/technical_glossary_de_en.csv"
)

# Define where the evaluation results will be saved.
OUTPUT_DIR = PROJECT_ROOT / "results/day12_terminology"

# Identify each translation example uniquely.
KEY_COLUMNS = ["language_pair", "domain", "sentence_id"]

# Define columns required for model evaluation.
REQUIRED_COLUMNS = KEY_COLUMNS + [
    "source",
    "reference",
    "prediction",
]


def load_glossary():
    """
    Load approved German-English terminology entries.
    """

    # Stop if the glossary does not exist.
    if not GLOSSARY_PATH.exists():
        raise FileNotFoundError(GLOSSARY_PATH)

    # Keep empty glossary cells as empty strings.
    glossary = pd.read_csv(
        GLOSSARY_PATH,
        dtype=str,
        keep_default_na=False,
    )

    # Verify that the checker can access its required fields.
    required = {
        "term_id",
        "source_de",
        "target_en",
        "alternatives_en",
    }

    missing = required - set(glossary.columns)

    if missing:
        raise ValueError(f"Missing glossary columns: {missing}")

    # Convert glossary records into dictionaries.
    return glossary.to_dict(orient="records")


def load_predictions():
    """
    Load and validate E0, E1 and E2 predictions.
    """

    # Store the validated datasets.
    datasets = {}

    for model, path in MODEL_FILES.items():

        # Stop if a required prediction file is missing.
        if not path.exists():
            raise FileNotFoundError(path)

        # Load the model's predictions.
        df = pd.read_csv(path)

        # Verify the expected CSV structure.
        missing = set(REQUIRED_COLUMNS) - set(df.columns)

        if missing:
            raise ValueError(
                f"{model} is missing columns: {missing}"
            )

        # Reject missing values in important columns.
        if df[REQUIRED_COLUMNS].isna().any().any():
            raise ValueError(
                f"{model} contains missing required values."
            )

        # Reject duplicated example identifiers.
        if df.duplicated(KEY_COLUMNS).any():
            raise ValueError(
                f"{model} contains duplicate example IDs."
            )

        # Keep the model name for the final report.
        df["model"] = model

        # Save the validated dataset.
        datasets[model] = df

        # Display the number of translations loaded.
        print(f"{model}: {len(df):,} predictions loaded")

    return datasets


def validate_alignment(datasets):
    """
    Confirm that all models evaluated identical source
    sentences and references.
    """

    # Use E0 as the reference dataset.
    baseline = datasets["E0"]

    # Build a comparable index using the example identifiers.
    baseline_index = baseline.set_index(KEY_COLUMNS)

    for model in ["E1", "E2"]:

        # Index the current model by the same identifiers.
        candidate = datasets[model].set_index(KEY_COLUMNS)

        # Compare example identifiers, regardless of row order.
        if set(baseline_index.index) != set(candidate.index):
            raise ValueError(
                f"{model} does not contain the same examples as E0."
            )

        # Align the candidate rows to the E0 ordering.
        candidate = candidate.reindex(baseline_index.index)

        # Verify identical source and reference translations.
        for column in ["source", "reference"]:

            if not baseline_index[column].equals(candidate[column]):
                raise ValueError(
                    f"{model} has mismatched {column} values."
                )

    print("Alignment verified: all models use identical examples.")


def evaluate_predictions(datasets, glossary):
    """
    Evaluate terminology and placeholder preservation.

    Only applicable checks are recorded:
    - Terminology: source contains an approved glossary term.
    - Placeholder: source contains at least one supported token.
    """

    # Collect individual check results.
    records = []

    # Process every model.
    for model, df in datasets.items():

        print(f"Evaluating {model}...")

        # Process each translation example.
        for row in df.itertuples(index=False):

            # Read the original source and model prediction.
            source = str(row.source)
            prediction = str(row.prediction)

            # Check approved technical terminology.
            term_results = check_terminology(
                source,
                prediction,
                glossary,
            )

            # Record each applicable terminology check.
            for result in term_results:

                records.append({
                    "model": model,
                    "domain": row.domain,
                    "sentence_id": row.sentence_id,
                    "check_type": "terminology",
                    "check_item": result["source_term"],
                    "passed": bool(result["passed"]),
                    "source": source,
                    "prediction": prediction,
                    "reference": row.reference,
                })

            # Evaluate protected placeholder tokens.
            placeholder_result = check_placeholders(
                source,
                prediction,
            )

            # Only score sentences containing placeholders.
            if placeholder_result["source_tokens"]:

                records.append({
                    "model": model,
                    "domain": row.domain,
                    "sentence_id": row.sentence_id,
                    "check_type": "placeholder",
                    "check_item": str(
                        placeholder_result["source_tokens"]
                    ),
                    "passed": bool(
                        placeholder_result["passed"]
                    ),
                    "source": source,
                    "prediction": prediction,
                    "reference": row.reference,
                })

            
            # Evaluate numeric preservation.
            number_result = check_numbers(
                source,
                prediction,
            )


            # ------------------------------------------------
            # DAY 12 — TECHNICAL UNIT PRESERVATION
            # ------------------------------------------------

            # Evaluate technical units in the translation.
            unit_result = check_units(
                source,
                prediction,
            )

            # Only evaluate sentences containing recognized units.
            # This avoids counting inapplicable sentences as passes.
            if unit_result["source_units"]:

                # Store the unit preservation result.
                records.append({
                    "model": model,
                    "domain": row.domain,
                    "sentence_id": row.sentence_id,
                    "check_type": "units",
                    "check_item": str(
                        unit_result["source_units"]
                    ),
                    "passed": bool(
                        unit_result["passed"]
                    ),
                    "source": source,
                    "prediction": prediction,
                    "reference": row.reference,
                })


            # Only score sentences containing source numbers.
            if number_result["source_numbers"]:

                # Store the numeric preservation outcome.
                records.append({
                    "model": model,
                    "domain": row.domain,
                    "sentence_id": row.sentence_id,
                    "check_type": "numbers",
                    "check_item": str(
                        number_result["source_numbers"]
                    ),
                    "passed": bool(
                        number_result["passed"]
                    ),
                    "source": source,
                    "prediction": prediction,
                    "reference": row.reference,
                })


            # ============================================================
            # DAY 12 — URL AND FILE-PATH PRESERVATION
            # ============================================================

            # Evaluate exact preservation of protected technical strings.
            for category, check_type in [
                ("urls", "url"),
                ("file_paths", "file_path"),
            ]:

                # Compare technical strings in the German source
                # against those in the English prediction.
                protected_result = check_protected_content(
                    source,
                    prediction,
                    category,
                )

                # Only evaluate applicable examples.
                # A sentence without a URL or path must not contribute
                # to that metric's denominator.
                if not protected_result["source_tokens"]:
                    continue

                # Add one sentence-level result per applicable category.
                # Multiple paths within one sentence count as one check:
                # all must be preserved for the check to pass.
                records.append({
                    "model": model,
                    "domain": row.domain,
                    "sentence_id": row.sentence_id,
                    "check_type": check_type,
                    "check_item": str(
                        protected_result["source_tokens"]
                    ),
                    "passed": bool(
                        protected_result["passed"]
                    ),
                    "source": source,
                    "prediction": prediction,
                    "reference": row.reference,
                })


    # Convert the evaluation results into a DataFrame.
    return pd.DataFrame(records)


def generate_summary(results):
    """
    Calculate compliance rates for each model,
    domain and evaluation category.
    """

    # Return a well-defined empty result if no checks apply.
    if results.empty:
        return pd.DataFrame(columns=[
            "model",
            "domain",
            "check_type",
            "total_checks",
            "passed_checks",
            "failed_checks",
            "compliance_percent",
        ])

    # Group results by model, domain and check type.
    summary = (
        results.groupby(
            ["model", "domain", "check_type"]
        )
        .agg(
            total_checks=("passed", "size"),
            passed_checks=("passed", "sum"),
        )
        .reset_index()
    )

    # Calculate the number of failed checks.
    summary["failed_checks"] = (
        summary["total_checks"]
        - summary["passed_checks"]
    )

    # Calculate compliance as a percentage.
    summary["compliance_percent"] = (
        summary["passed_checks"]
        / summary["total_checks"]
        * 100
    ).round(2)

    return summary


def main():
    """
    Execute the complete Day 12 evaluation pipeline.
    """

    # Create the output directory if necessary.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load the approved terminology list.
    glossary = load_glossary()

    print(f"Glossary entries: {len(glossary)}")

    # Load the model predictions.
    datasets = load_predictions()

    # Verify that model comparisons are fair.
    validate_alignment(datasets)

    # Evaluate terminology and placeholders.
    results = evaluate_predictions(datasets, glossary)

    # Calculate the aggregated compliance metrics.
    summary = generate_summary(results)

    # Extract examples flagged for manual inspection.
    flagged = results.loc[~results["passed"]].copy()

    # Save the individual evaluation records.
    results.to_csv(
        OUTPUT_DIR / "terminology_results.csv",
        index=False,
    )

    # Save all flagged examples.
    flagged.to_csv(
        OUTPUT_DIR / "flagged_examples.csv",
        index=False,
    )

    # Save the aggregated model comparison.
    summary.to_csv(
        OUTPUT_DIR / "summary_metrics.csv",
        index=False,
    )

    # Display the final evaluation summary.
    print("\nDAY 12 EVALUATION SUMMARY")

    if summary.empty:
        print("No applicable terminology or placeholder checks.")
    else:
        print(summary.to_string(index=False))

    # Display output locations.
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print(f"Total checks: {len(results):,}")
    print(f"Flagged checks: {len(flagged):,}")


# Run the evaluation only when executed as a script.
if __name__ == "__main__":
    main()
