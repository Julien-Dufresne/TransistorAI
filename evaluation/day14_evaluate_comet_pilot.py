from pathlib import Path
import pandas as pd
import sacrebleu

from day11_evaluate_comet import calculate_comet
from comet import download_model, load_from_checkpoint

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/day14_llm_baseline"
INPUT = OUTPUT / "pilot_aligned_predictions.csv"

MODEL_NAME = "Unbabel/wmt22-comet-da"

def main():
    df = pd.read_csv(INPUT, keep_default_na=False)

    expected_models = [
        "E0_baseline",
        "E1_technical",
        "E2_mixed",
        "E3_Qwen",
    ]

    assert len(df) == 400, f"Expected 400 rows, got {len(df)}"

    print("Loading COMET...", flush=True)
    checkpoint = download_model(MODEL_NAME)
    model = load_from_checkpoint(checkpoint)

    summary = []
    segments = []

    for model_name in expected_models:
        for domain in ["technical", "general"]:
            subset = df[
                (df["model"] == model_name)
                & (df["domain"] == domain)
            ].copy()

            assert len(subset) == 50, (
                f"{model_name}/{domain}: expected 50 rows"
            )

            print(
                f"Evaluating {model_name}/{domain}...",
                flush=True,
            )

            comet_score, segment_scores = calculate_comet(
                model,
                subset,
                batch_size=8,
                gpus=0,
            )

            assert len(segment_scores) == len(subset)

            subset["comet_score"] = segment_scores
            segments.append(subset)

            bleu = sacrebleu.corpus_bleu(
                subset["prediction"].tolist(),
                [subset["reference"].tolist()],
            ).score

            chrf = sacrebleu.corpus_chrf(
                subset["prediction"].tolist(),
                [subset["reference"].tolist()],
            ).score

            summary.append({
                "model": model_name,
                "domain": domain,
                "n": len(subset),
                "BLEU": round(bleu, 3),
                "chrF": round(chrf, 3),
                "COMET": round(comet_score, 4),
            })

            # Save progress after each model/domain.
            pd.DataFrame(summary).to_csv(
                OUTPUT / "pilot_comet_scores.csv",
                index=False,
            )

            pd.concat(segments, ignore_index=True).to_csv(
                OUTPUT / "pilot_comet_segments.csv",
                index=False,
            )

            print(
                f"{model_name}/{domain}: "
                f"COMET={comet_score:.4f}",
                flush=True,
            )

    print("\nFOUR-MODEL COMET COMPARISON\n")
    print(pd.DataFrame(summary).to_string(index=False))

    print("\nSaved:")
    print(OUTPUT / "pilot_comet_scores.csv")
    print(OUTPUT / "pilot_comet_segments.csv")


if __name__ == "__main__":
    main()
