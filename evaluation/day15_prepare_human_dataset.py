from pathlib import Path
import pandas as pd

SEED = 42
PER_DOMAIN = 100
OUTPUT = Path("data/human_evaluation/preference_dataset.csv")

FILES = {
    "E0": "results/day11_evaluation/e0_generation/predictions.csv",
    "E1": "results/day8-de-en-v2/e1_evaluation/predictions.csv",
    "E2": "results/day8-de-en-v2/e2_evaluation/predictions.csv",
    "E3": "results/day14_llm_baseline/predictions.csv",
}

KEYS = ["language_pair", "domain", "dataset", "sentence_id"]
TEXT = ["source", "reference"]

def main():
    frames = {}

    for model, path in FILES.items():
        df = pd.read_csv(path, keep_default_na=False)

        if df.duplicated(KEYS).any():
            raise ValueError(f"Duplicate keys in {model}")

        if df[TEXT + ["prediction"]].apply(
            lambda c: c.astype(str).str.strip().eq("")
        ).any().any():
            raise ValueError(f"Empty text in {model}")

        frames[model] = df

    combined = frames["E0"][KEYS + TEXT + ["prediction"]].rename(
        columns={"prediction": "translation_E0"}
    )

    for model in ["E1", "E2", "E3"]:
        other = frames[model][KEYS + TEXT + ["prediction"]].rename(
            columns={
                "source": f"source_{model}",
                "reference": f"reference_{model}",
                "prediction": f"translation_{model}",
            }
        )

        combined = combined.merge(
            other,
            on=KEYS,
            how="inner",
            validate="one_to_one",
        )

        if not combined["source"].equals(combined[f"source_{model}"]):
            raise ValueError(f"Source mismatch for {model}")

        if not combined["reference"].equals(
            combined[f"reference_{model}"]
        ):
            raise ValueError(f"Reference mismatch for {model}")

        combined = combined.drop(
            columns=[f"source_{model}", f"reference_{model}"]
        )

    if len(combined) != len(frames["E0"]):
        raise ValueError("Incomplete model alignment")

    # ============================================================
    # FILTER LOW-INFORMATION EXAMPLES
    # ============================================================

    translation_columns = [
        "translation_E0",
        "translation_E1",
        "translation_E2",
        "translation_E3",
    ]

    # Count German words.
    combined["source_words"] = (
        combined["source"].astype(str).str.split().str.len()
    )

    # Count distinct translations for each sentence.
    combined["unique_outputs"] = (
        combined[translation_columns].nunique(axis=1)
    )

    # Exclude very short sentences and examples where models
    # produce fewer than three different translations.
    eligible = combined[
        (combined["source_words"] >= 5)
        & (combined["unique_outputs"] >= 3)
    ].copy()

    # Exclude German sentences identical to all four outputs.
    identical_to_source = eligible[translation_columns].eq(
        eligible["source"], axis=0
    ).all(axis=1)

    eligible = eligible[~identical_to_source].copy()

    # Remove repeated German source sentences.
    eligible = eligible.sort_values(
        ["domain", "dataset", "sentence_id"],
        key=lambda column: column.astype(str),
    ).drop_duplicates(subset=["source"])

    print("\nELIGIBLE SENTENCES BY DOMAIN")
    print(eligible["domain"].value_counts())

    samples = []

    for domain in ["technical", "general"]:
        subset = eligible[eligible["domain"] == domain]

        if len(subset) < PER_DOMAIN:
            raise ValueError(
                f"Not enough eligible {domain} sentences: "
                f"{len(subset)} available, {PER_DOMAIN} required"
            )

        samples.append(
            subset.sample(n=PER_DOMAIN, random_state=SEED)
        )

    selected = pd.concat(samples, ignore_index=True)
    selected = selected.sample(
        frac=1, random_state=SEED
    ).reset_index(drop=True)

    selected.insert(
        0,
        "evaluation_id",
        [f"HE{i:04d}" for i in range(1, len(selected) + 1)],
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(OUTPUT, index=False)

    print("Saved:", OUTPUT)
    print("Total:", len(selected))
    print("\nDomain distribution:")
    print(selected["domain"].value_counts())
    print("\nColumns:")
    print(selected.columns.tolist())


if __name__ == "__main__":
    main()
