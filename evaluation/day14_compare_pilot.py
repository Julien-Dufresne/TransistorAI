from pathlib import Path
import pandas as pd
import sacrebleu

FILES = {
    "E0_baseline": "results/day11_evaluation/e0_generation/predictions.csv",
    "E1_technical": "results/day8-de-en-v2/e1_evaluation/predictions.csv",
    "E2_mixed": "results/day8-de-en-v2/e2_evaluation/predictions.csv",
    "E3_Qwen": "results/day14_llm_baseline/pilot_predictions.csv",
}

KEYS = [
    "language_pair", "domain", "dataset",
    "sentence_id", "source", "reference",
]

output_dir = Path("results/day14_llm_baseline")
pilot = pd.read_csv(FILES["E3_Qwen"], keep_default_na=False)

assert len(pilot) == 100
assert not pilot.duplicated(KEYS).any()

results = []
aligned_outputs = []

for model_name, path in FILES.items():
    df = pd.read_csv(path, keep_default_na=False)

    assert not df.duplicated(KEYS).any(), (
        f"Duplicate evaluation keys: {model_name}"
    )

    matched = pilot[KEYS].merge(
        df[KEYS + ["prediction"]],
        on=KEYS,
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    assert (matched["_merge"] == "both").all(), (
        f"Missing aligned examples for {model_name}"
    )

    assert not matched["prediction"].str.strip().eq("").any(), (
        f"Empty predictions for {model_name}"
    )

    matched["model"] = model_name
    aligned_outputs.append(
        matched.drop(columns=["_merge"])
    )

    for domain in ["technical", "general", "all"]:
        subset = (
            matched if domain == "all"
            else matched[matched["domain"] == domain]
        )

        predictions = subset["prediction"].tolist()
        references = subset["reference"].tolist()

        bleu = sacrebleu.corpus_bleu(
            predictions, [references]
        ).score

        chrf = sacrebleu.corpus_chrf(
            predictions, [references]
        ).score

        results.append({
            "model": model_name,
            "domain": domain,
            "n": len(subset),
            "BLEU": round(bleu, 3),
            "chrF": round(chrf, 3),
        })

scores = pd.DataFrame(results)
scores.to_csv(output_dir / "pilot_comparison.csv", index=False)

pd.concat(aligned_outputs, ignore_index=True).to_csv(
    output_dir / "pilot_aligned_predictions.csv",
    index=False,
)

print("\nDAY 14 — FOUR-MODEL PILOT COMPARISON\n")
print(scores.to_string(index=False))

print("\nSaved comparison and aligned predictions.")
