
"""
Day 17 — Compare human preferences with BLEU, chrF and COMET.

Uses frozen human evaluation data, existing COMET segment
scores, and SacreBLEU 2.6.0.

Does not modify votes or existing evaluation files.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
from sacrebleu.metrics import BLEU, CHRF


DATASET = Path("data/human_evaluation/preference_dataset.csv")
DATABASE = Path("data/human_evaluation/preference_votes.sqlite3")
OUTPUT = Path("results/day17_human_metrics")

MODELS = ("E0", "E1", "E2", "E3")
METRICS = ("bleu", "chrf", "comet")
KEYS = ["domain", "dataset", "sentence_id"]

COMET_FILES = {
    "E0": [
        "results/day11_evaluation/e0_comet/E0_general_segments.csv",
        "results/day11_evaluation/e0_comet/E0_technical_segments.csv",
    ],
    "E1": [
        "results/day11_evaluation/E1_general_segments.csv",
        "results/day11_evaluation/E1_technical_segments.csv",
    ],
    "E2": [
        "results/day11_evaluation/E2_general_segments.csv",
        "results/day11_evaluation/E2_technical_segments.csv",
    ],
    "E3": [
        "results/day14_llm_baseline/full_comet_segments.csv",
    ],
}


def load_comet(model):
    frames = [
        pd.read_csv(path)
        for path in COMET_FILES[model]
    ]
    segments = pd.concat(frames, ignore_index=True)

    if segments.duplicated(KEYS).any():
        raise ValueError(f"Duplicate COMET keys for {model}")

    return segments


def build_sentence_metrics(human):
    bleu = BLEU(effective_order=True)
    chrf = CHRF()

    records = []

    for model in MODELS:
        segments = load_comet(model)

        aligned = human.merge(
            segments,
            on=KEYS,
            how="left",
            validate="one_to_one",
            indicator=True,
            suffixes=("_human", "_comet"),
        )

        if not (aligned["_merge"] == "both").all():
            raise ValueError(f"Missing COMET rows for {model}")

        for _, row in aligned.iterrows():
            prediction = row[f"translation_{model}"]
            reference = row["reference_human"]

            for field, expected in [
                ("prediction", prediction),
                ("source_comet", row["source_human"]),
                ("reference_comet", reference),
            ]:
                if row[field] != expected:
                    raise ValueError(
                        f"{model} alignment mismatch: "
                        f"{row['evaluation_id']} ({field})"
                    )

            if pd.isna(row["comet_score"]):
                raise ValueError("Missing COMET score")

            records.append({
                "evaluation_id": row["evaluation_id"],
                "domain": row["domain"],
                "model": model,
                "source": row["source_human"],
                "reference": reference,
                "translation": prediction,
                "bleu": bleu.sentence_score(
                    prediction, [reference]
                ).score,
                "chrf": chrf.sentence_score(
                    prediction, [reference]
                ).score,
                "comet": float(row["comet_score"]),
            })

    result = pd.DataFrame(records)

    if len(result) != len(human) * len(MODELS):
        raise ValueError("Unexpected metric row count")

    return result


def load_votes():
    if not DATABASE.exists():
        raise FileNotFoundError(DATABASE)

    with sqlite3.connect(
        f"file:{DATABASE.resolve()}?mode=ro",
        uri=True,
    ) as conn:
        votes = pd.read_sql_query(
            """
            SELECT participant_id, evaluation_id,
                   domain, selected_text, selected_models
            FROM votes
            """,
            conn,
        )

    if votes.duplicated(
        ["participant_id", "evaluation_id"]
    ).any():
        raise ValueError("Duplicate human votes")

    return votes


def compare_human_metrics(scores, votes):
    records = []

    for evaluation_id, group in scores.groupby("evaluation_id"):
        sentence_votes = votes[
            votes["evaluation_id"] == evaluation_id
        ]

        if sentence_votes.empty:
            continue

        # Group identical translations before ranking.
        # Each distinct translation is one candidate.
        candidates = (
            group.groupby("translation", as_index=False)
            .agg(
                bleu=("bleu", "mean"),
                chrf=("chrf", "mean"),
                comet=("comet", "mean"),
            )
        )

        # Models producing identical text should normally
        # have identical scores. Check rather than assume.
        for metric in METRICS:
            spread = group.groupby("translation")[metric].agg(
                lambda values: values.max() - values.min()
            )

            if (spread > 1e-5).any():
                raise ValueError(
                    f"Identical translations have differing "
                    f"{metric} scores: {evaluation_id}"
                )

        for metric in METRICS:
            best_score = candidates[metric].max()

            # Floating-point tolerance for genuine score ties.
            tolerance = 1e-8 if metric != "comet" else 1e-6

            winners = candidates[
                (candidates[metric] - best_score).abs()
                <= tolerance
            ]["translation"].tolist()

            winning_models = sorted(
                group[
                    group["translation"].isin(winners)
                ]["model"].unique()
            )

            for _, vote in sentence_votes.iterrows():
                selected = vote["selected_text"]
                selected_models = json.loads(
                    vote["selected_models"]
                )

                if selected not in candidates["translation"].values:
                    raise ValueError(
                        f"Unknown selected text: {evaluation_id}"
                    )

                agreement = selected in winners

                records.append({
                    "evaluation_id": evaluation_id,
                    "domain": vote["domain"],
                    "participant_id": vote["participant_id"],
                    "metric": metric,
                    "human_selected_text": selected,
                    "human_selected_models": json.dumps(
                        selected_models
                    ),
                    "metric_winning_models": json.dumps(
                        winning_models
                    ),
                    "metric_winning_texts": json.dumps(
                        winners, ensure_ascii=False
                    ),
                    "metric_best_score": best_score,
                    "metric_tie_count": len(winners),
                    "strict_agreement": int(agreement),
                    "fractional_agreement": (
                        1 / len(winners) if agreement else 0
                    ),
                })

    return pd.DataFrame(records)


def summarize(comparison):
    records = []

    for metric in METRICS:
        metric_df = comparison[
            comparison["metric"] == metric
        ]

        for domain in ["overall", "general", "technical"]:
            subset = (
                metric_df
                if domain == "overall"
                else metric_df[metric_df["domain"] == domain]
            )

            if subset.empty:
                continue

            records.append({
                "metric": metric,
                "domain": domain,
                "votes": len(subset),
                "unique_sentences": subset[
                    "evaluation_id"
                ].nunique(),
                "strict_agreement_percent": round(
                    subset["strict_agreement"].mean() * 100,
                    2,
                ),
                "fractional_agreement_percent": round(
                    subset["fractional_agreement"].mean() * 100,
                    2,
                ),
                "metric_tied_votes": int(
                    (subset["metric_tie_count"] > 1).sum()
                ),
            })

    return pd.DataFrame(records)


def main():
    human = pd.read_csv(DATASET)

    if human.duplicated(KEYS).any():
        raise ValueError("Duplicate dataset keys")

    if human["evaluation_id"].duplicated().any():
        raise ValueError("Duplicate evaluation IDs")

    votes = load_votes()

    if not set(votes["evaluation_id"]).issubset(
        human["evaluation_id"]
    ):
        raise ValueError("Unknown evaluation IDs in votes")

    domain_map = human.set_index("evaluation_id")["domain"]

    if not votes.apply(
        lambda row: (
            row["domain"]
            == domain_map.loc[row["evaluation_id"]]
        ),
        axis=1,
    ).all():
        raise ValueError("Human vote domain mismatch")

    print("Calculating sentence-level metrics...")
    scores = build_sentence_metrics(human)

    print("Comparing human selections...")
    comparison = compare_human_metrics(scores, votes)

    summary = summarize(comparison)

    disagreements = comparison[
        comparison["strict_agreement"] == 0
    ].copy()

    OUTPUT.mkdir(parents=True, exist_ok=True)

    outputs = {
        "sentence_level_metrics.csv": scores,
        "human_metric_comparison.csv": comparison,
        "metric_agreement_summary.csv": summary,
        "metric_disagreements.csv": disagreements,
    }

    for filename, frame in outputs.items():
        path = OUTPUT / filename
        frame.to_csv(path, index=False)
        print(f"Saved: {path} ({len(frame)} rows)")

    print("\n=== DAY 17 RESULTS ===")
    print("Sentences:", len(human))
    print("Model–sentence scores:", len(scores))
    print("Human votes:", len(votes))
    print("Metric comparisons:", len(comparison))

    print("\nHUMAN VS AUTOMATIC METRICS")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
