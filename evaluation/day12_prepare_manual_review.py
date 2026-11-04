
"""
Day 12 — Prepare a reproducible manual review sample.

Select flagged terminology and placeholder checks
for human validation.

The script never changes the original evaluation results.
"""

# Import Path to manage project file locations.
from pathlib import Path

# Import pandas for sampling and CSV processing.
import pandas as pd

# Define the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the directory containing Day 12 results.
RESULTS_DIR = PROJECT_ROOT / "results/day12_terminology"

# Define the input and output files.
INPUT_FILE = RESULTS_DIR / "flagged_examples.csv"
OUTPUT_FILE = RESULTS_DIR / "manual_review.csv"

# Set a reproducible random seed.
RANDOM_SEED = 42

# Define the maximum number of checks to review.
SAMPLE_SIZE = 50


def main():
    """
    Select a balanced sample and prepare annotation columns.
    """

    # Load the flagged evaluation checks.
    df = pd.read_csv(INPUT_FILE)

    # Verify that flagged checks are available.
    if df.empty:
        print("No flagged checks available.")
        return

    # Define the grouping dimensions.
    group_columns = ["model", "domain", "check_type"]

    # Group the flags by model, domain and check category.
    groups = [
        group
        for _, group in df.groupby(group_columns, sort=True)
    ]

    # Determine how many rows to sample from each group.
    target = min(SAMPLE_SIZE, len(df))

    # Allocate an approximately equal initial quota.
    quota = max(1, target // len(groups))

    # Sample each group reproducibly.
    selected = pd.concat(
        [
            group.sample(
                n=min(len(group), quota),
                random_state=RANDOM_SEED,
            )
            for group in groups
        ],
        ignore_index=False,
    )

    # Fill unused places from the remaining flagged checks.
    remaining = df.drop(index=selected.index)

    # Calculate how many more checks are needed.
    missing = target - len(selected)

    if missing > 0:

        # Sample additional checks without replacement.
        extra = remaining.sample(
            n=min(missing, len(remaining)),
            random_state=RANDOM_SEED,
        )

        # Combine the initial sample and additional checks.
        selected = pd.concat([selected, extra])

    # Sort selected checks for a convenient review workflow.
    selected = selected.sort_values(
        group_columns + ["sentence_id"]
    ).reset_index(drop=True)

    # Assign a unique identifier to each review item.
    selected.insert(
        0,
        "review_id",
        range(1, len(selected) + 1),
    )

    # Add empty annotation columns for manual review.
    selected["manual_label"] = ""
    selected["error_category"] = ""
    selected["review_notes"] = ""

    # Save the review template.
    selected.to_csv(OUTPUT_FILE, index=False)

    # Display a short sampling summary.
    print(f"Total flagged checks: {len(df)}")
    print(f"Selected for review: {len(selected)}")

    print("\nSAMPLE DISTRIBUTION:")
    print(
        selected.groupby(group_columns)
        .size()
        .to_string()
    )

    print(f"\nSaved to: {OUTPUT_FILE}")


# Execute the script when run directly.
if __name__ == "__main__":
    main()
