
# ============================================================
# TRANSISTORAI — DAY 11
# BLEU, chrF AND COMET EVALUATION
# ============================================================

# Import standard Python libraries.
import argparse
import json
import platform
import sys
import time
from pathlib import Path

# Import data processing and evaluation libraries.
import pandas as pd
import sacrebleu
import torch
from comet import download_model, load_from_checkpoint

# Import version metadata to document the environment.
from importlib.metadata import version


# Define the project root based on this script's location.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Define the columns that uniquely identify each test example.
KEY_COLUMNS = ["domain", "dataset", "sentence_id"]

# Define the columns required for evaluation.
REQUIRED_COLUMNS = [
    "language_pair",
    "domain",
    "dataset",
    "sentence_id",
    "source",
    "reference",
    "prediction",
]


def load_config(config_path):
    """Read the Day 11 JSON configuration."""

    # Resolve relative paths against the project root.
    path = Path(config_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    # Read and parse the configuration.
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def load_predictions(path, system_name, config):
    """Load and validate one system's saved predictions."""

    # Resolve the CSV path.
    csv_path = PROJECT_ROOT / path

    # Fail clearly if the prediction file is missing.
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path)

    # Read the complete prediction file.
    df = pd.read_csv(csv_path)

    # Verify all required columns are available.
    missing = set(REQUIRED_COLUMNS) - set(df.columns)

    if missing:
        raise ValueError(
            f"{system_name}: missing columns {sorted(missing)}"
        )

    # Keep only the requested translation direction and domains.
    df = df[
        (df["language_pair"] == config["language_pair"])
        & (
            df["domain"].isin(
                config["evaluation"]["domains"]
            )
        )
    ].copy()

    # Reject duplicate example identifiers.
    if df.duplicated(KEY_COLUMNS).any():
        raise ValueError(
            f"{system_name}: duplicate example identifiers"
        )

    # Reject missing or blank source, reference or prediction.
    for column in ["source", "reference", "prediction"]:

        if df[column].isna().any():
            raise ValueError(
                f"{system_name}: missing {column} values"
            )

        if df[column].astype(str).str.strip().eq("").any():
            raise ValueError(
                f"{system_name}: empty {column} values"
            )

    # Sort examples deterministically.
    return df.sort_values(KEY_COLUMNS).reset_index(drop=True)


def validate_alignment(dataframes):
    """Verify all systems use identical held-out examples."""

    # Use the first system as the comparison reference.
    reference_name = next(iter(dataframes))
    reference_df = dataframes[reference_name]

    # Extract the identifying fields and original text.
    columns = KEY_COLUMNS + ["source", "reference"]

    # Compare every other system with the reference system.
    for name, df in dataframes.items():

        # Require exactly the same example identities and texts.
        if not reference_df[columns].equals(df[columns]):
            raise ValueError(
                f"Alignment failed: {reference_name} vs {name}"
            )

    print("Alignment verified across all systems.", flush=True)


def calculate_lexical_metrics(df):
    """Calculate corpus-level BLEU and chrF scores."""

    # Extract translation predictions.
    predictions = df["prediction"].tolist()

    # SacreBLEU expects a list of reference streams.
    references = [df["reference"].tolist()]

    # Calculate corpus BLEU using SacreBLEU.
    bleu = sacrebleu.corpus_bleu(
        predictions,
        references,
    ).score

    # Calculate corpus chrF using SacreBLEU.
    chrf = sacrebleu.corpus_chrf(
        predictions,
        references,
    ).score

    return bleu, chrf



def calculate_comet(model, df, batch_size, gpus):
    """
    Calculate reference-based COMET translation scores.

    Use a DataLoader worker to avoid the multiprocessing
    compatibility error encountered with num_workers=0.
    """

    # Convert each translation example into COMET's format.
    samples = [
        {
            "src": row.source,
            "mt": row.prediction,
            "ref": row.reference,
        }
        for row in df.itertuples(index=False)
    ]

    # Run COMET inference using one DataLoader worker.
    # This avoids the previous num_workers=0 configuration.
    # CPU execution is retained for compatibility on your Mac.
    output = model.predict(
        samples,
        batch_size=batch_size,
        gpus=gpus,
        num_workers=1,
    )

    # Return the corpus score and individual segment scores.
    return (
        float(output.system_score),
        [float(score) for score in output.scores],
    )



def main():
    """Execute the complete Day 11 evaluation."""

    # Read the configuration argument.
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        required=True,
        help="Path to Day 11 configuration JSON",
    )

    args = parser.parse_args()

    # Load the experiment configuration.
    config = load_config(args.config)

    # Create the output directory.
    output_dir = (
        PROJECT_ROOT / config["paths"]["output_dir"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Display the evaluation mode.
    smoke_test = config["evaluation"]["smoke_test"]

    print("DAY 11 — COMET EVALUATION", flush=True)
    print(f"Smoke test: {smoke_test}", flush=True)

    # Load each configured system's predictions.
    dataframes = {}

    for system in config["evaluation"]["systems"]:

        # Resolve the configured path key.
        path_key = f"{system.lower()}_predictions"

        # Require an explicit prediction path.
        if path_key not in config["paths"]:
            raise ValueError(
                f"Missing configuration path: {path_key}"
            )

        # Load and validate the prediction file.
        dataframes[system] = load_predictions(
            config["paths"][path_key],
            system,
            config,
        )

        print(
            f"{system}: {len(dataframes[system])} rows",
            flush=True,
        )

    # Validate alignment before selecting smoke-test examples.
    if config["evaluation"]["require_identical_examples"]:
        validate_alignment(dataframes)

    # Restrict each domain to the same number of examples.
    if smoke_test:

        sample_count = config["evaluation"]["smoke_test_samples"]

        for system, df in dataframes.items():

            # Select the first N examples within each domain.
            dataframes[system] = (
                df.groupby("domain", group_keys=False)
                .head(sample_count)
                .reset_index(drop=True)
            )

    # Download or reuse the cached COMET checkpoint.
    model_name = config["comet"]["model_name"]

    print(f"Loading COMET: {model_name}", flush=True)

    checkpoint_path = download_model(model_name)

    # Load the reference-based COMET model.
    model = load_from_checkpoint(checkpoint_path)

    print("COMET model ready.", flush=True)

    # Store one summary row per system and domain.
    summary_rows = []

    # Evaluate every configured system.
    for system, df in dataframes.items():

        # Evaluate each domain independently.
        for domain in config["evaluation"]["domains"]:

            # Select only the requested domain.
            subset = df[df["domain"] == domain].copy()

            # Skip domains with no available examples.
            if subset.empty:
                print(
                    f"Skipping {system}/{domain}: no examples",
                    flush=True,
                )
                continue

            print(
                f"Evaluating {system}/{domain}: "
                f"{len(subset)} examples",
                flush=True,
            )

            # Start the runtime measurement.
            start_time = time.perf_counter()

            # Calculate corpus BLEU and chrF.
            bleu, chrf = calculate_lexical_metrics(subset)

            # Calculate COMET system and segment scores.
            comet_score, segment_scores = calculate_comet(
                model,
                subset,
                batch_size=config["comet"]["batch_size"],
                gpus=config["comet"]["gpus"],
            )

            # Measure total evaluation time.
            runtime = time.perf_counter() - start_time

            # Verify that COMET returned one score per example.
            if len(segment_scores) != len(subset):
                raise ValueError(
                    f"{system}/{domain}: COMET score count mismatch"
                )

            # Attach segment-level COMET scores.
            subset["comet_score"] = segment_scores

            # Save individual prediction and score records.
            subset.to_csv(
                output_dir / f"{system}_{domain}_segments.csv",
                index=False,
            )

            # Save the corpus-level evaluation summary.
            summary_rows.append(
                {
                    "system": system,
                    "domain": domain,
                    "examples": len(subset),
                    "bleu": bleu,
                    "chrf": chrf,
                    "comet": comet_score,
                    "runtime_seconds": runtime,
                    "status": "completed",
                }
            )

            # Save intermediate results after every domain.
            pd.DataFrame(summary_rows).to_csv(
                output_dir / "comparison.csv",
                index=False,
            )

            print(
                f"{system}/{domain} completed: "
                f"BLEU={bleu:.3f}, "
                f"chrF={chrf:.3f}, "
                f"COMET={comet_score:.4f}, "
                f"runtime={runtime:.1f}s",
                flush=True,
            )

    # Save reproducibility metadata.
    metadata = {
        "experiment": config["experiment_name"],
        "language_pair": config["language_pair"],
        "comet_model": model_name,
        "comet_revision": config["comet"]["model_revision"],
        "checkpoint_path": str(checkpoint_path),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "sacrebleu": version("sacrebleu"),
        "unbabel_comet": version("unbabel-comet"),
        "smoke_test": smoke_test,
        "systems_evaluated": list(dataframes.keys()),
    }

    # Save environment and metric metadata.
    with (output_dir / "environment.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2, default=str)

    print("\nDay 11 evaluation completed.", flush=True)


# Execute the script when launched directly.
if __name__ == "__main__":
    main()
