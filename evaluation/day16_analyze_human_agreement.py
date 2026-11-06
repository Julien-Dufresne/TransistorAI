
"""
Day 16 — Human preference agreement analysis.

Reads:
    data/human_evaluation/preference_dataset.csv
    data/human_evaluation/preference_votes.sqlite3

Writes:
    results/day16_human_agreement/*.csv

Does not modify the SQLite database.
"""

import json
import sqlite3
from itertools import combinations
from pathlib import Path

import pandas as pd

from evaluation.day15_human_preference import load_dataset
from human_preference_page import get_shared_evaluation_ids


DB_PATH = Path(
    "data/human_evaluation/preference_votes.sqlite3"
)

OUTPUT_DIR = Path("results/day16_human_agreement")

MODELS = ["E0", "E1", "E2", "E3"]


def load_votes():
    if not DB_PATH.exists():
        raise FileNotFoundError(DB_PATH)

    # Open SQLite in read-only mode.
    with sqlite3.connect(
        f"file:{DB_PATH.resolve()}?mode=ro",
        uri=True,
    ) as conn:
        votes = pd.read_sql_query(
            """
            SELECT
                participant_id,
                evaluation_id,
                domain,
                selected_option,
                selected_text,
                selected_models,
                option_mapping,
                voted_at
            FROM votes
            """,
            conn,
        )

    if votes.empty:
        raise ValueError("No human evaluation votes found.")

    if votes.duplicated(
        ["participant_id", "evaluation_id"]
    ).any():
        raise ValueError("Duplicate participant/sentence votes.")

    votes["models"] = votes["selected_models"].apply(
        lambda raw: sorted(set(json.loads(raw)))
    )

    for models in votes["models"]:
        if not models or not set(models).issubset(MODELS):
            raise ValueError(f"Invalid selected models: {models}")

    return votes


def calculate_pairwise_agreement(votes, dataset):
    records = []

    for evaluation_id, group in votes.groupby("evaluation_id"):
        if len(group) < 2:
            continue

        domain = dataset.loc[evaluation_id, "domain"]

        for (_, a), (_, b) in combinations(
            group.iterrows(), 2
        ):
            exact = (
                a["selected_text"].strip()
                == b["selected_text"].strip()
            )

            model_overlap = bool(
                set(a["models"]) & set(b["models"])
            )

            records.append({
                "evaluation_id": evaluation_id,
                "domain": domain,
                "participant_a": a["participant_id"],
                "participant_b": b["participant_id"],
                "exact_agreement": int(exact),
                "model_overlap": int(model_overlap),
            })

    return pd.DataFrame(
        records,
        columns=[
            "evaluation_id",
            "domain",
            "participant_a",
            "participant_b",
            "exact_agreement",
            "model_overlap",
        ],
    )


def agreement_summary(pairwise):
    records = []

    for domain in ["overall", "general", "technical"]:
        group = (
            pairwise
            if domain == "overall"
            else pairwise[pairwise["domain"] == domain]
        )

        records.append({
            "domain": domain,
            "participant_pairs": len(group),
            "unique_sentences": group["evaluation_id"].nunique(),
            "exact_agreements": int(
                group["exact_agreement"].sum()
            ),
            "exact_agreement_percent": (
                round(group["exact_agreement"].mean() * 100, 2)
                if len(group) else None
            ),
            "model_overlap_percent": (
                round(group["model_overlap"].mean() * 100, 2)
                if len(group) else None
            ),
        })

    return pd.DataFrame(records)


def model_preferences(votes):
    records = []

    for _, vote in votes.iterrows():
        models = vote["models"]
        fractional_credit = 1 / len(models)

        for model in models:
            records.append({
                "participant_id": vote["participant_id"],
                "evaluation_id": vote["evaluation_id"],
                "domain": vote["domain"],
                "model": model,
                "full_selection": 1,
                "fractional_credit": fractional_credit,
                "grouped_selection": int(len(models) > 1),
            })

    expanded = pd.DataFrame(records)

    return (
        expanded.groupby(
            ["participant_id", "domain", "model"],
            as_index=False,
        )
        .agg(
            full_selections=("full_selection", "sum"),
            fractional_selections=("fractional_credit", "sum"),
            grouped_selections=("grouped_selection", "sum"),
        )
        .sort_values(
            ["participant_id", "domain", "model"]
        )
    )


def disagreement_examples(votes, dataset, pairwise):
    records = []

    disagreements = pairwise[
        pairwise["exact_agreement"] == 0
    ]

    vote_index = votes.set_index(
        ["participant_id", "evaluation_id"]
    )

    for _, pair in disagreements.iterrows():
        evaluation_id = pair["evaluation_id"]

        a = vote_index.loc[
            (pair["participant_a"], evaluation_id)
        ]
        b = vote_index.loc[
            (pair["participant_b"], evaluation_id)
        ]

        sentence = dataset.loc[evaluation_id]

        records.append({
            "evaluation_id": evaluation_id,
            "domain": pair["domain"],
            "source": sentence["source"],
            "reference": sentence["reference"],
            "participant_a": pair["participant_a"],
            "participant_b": pair["participant_b"],
            "selected_text_a": a["selected_text"],
            "selected_text_b": b["selected_text"],
            "selected_models_a": json.dumps(a["models"]),
            "selected_models_b": json.dumps(b["models"]),
        })

    return pd.DataFrame(
        records,
        columns=[
            "evaluation_id",
            "domain",
            "source",
            "reference",
            "participant_a",
            "participant_b",
            "selected_text_a",
            "selected_text_b",
            "selected_models_a",
            "selected_models_b",
        ],
    )


def shared_subset_coverage(votes, dataset):
    shared_ids = get_shared_evaluation_ids(
        dataset.reset_index(drop=True)
    )

    records = []

    for evaluation_id in shared_ids:
        group = votes[
            votes["evaluation_id"] == evaluation_id
        ]

        selected_texts = group["selected_text"].tolist()

        records.append({
            "evaluation_id": evaluation_id,
            "domain": dataset.loc[evaluation_id, "domain"],
            "participant_count": group["participant_id"].nunique(),
            "vote_count": len(group),
            "unique_selected_texts": len(set(selected_texts)),
            "unanimous": (
                len(group) >= 2
                and len(set(selected_texts)) == 1
            ),
        })

    return pd.DataFrame(records)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    votes = load_votes()

    dataset = load_dataset().set_index(
        "evaluation_id", drop=False
    )

    unknown_ids = (
        set(votes["evaluation_id"])
        - set(dataset.index)
    )

    if unknown_ids:
        raise ValueError(
            f"Votes reference unknown sentences: {unknown_ids}"
        )

    mismatched_domains = votes.apply(
        lambda row: row["domain"]
        != dataset.loc[row["evaluation_id"], "domain"],
        axis=1,
    )

    if mismatched_domains.any():
        raise ValueError("Vote/domain mismatch detected.")

    pairwise = calculate_pairwise_agreement(
        votes, dataset
    )

    summary = agreement_summary(pairwise)
    preferences = model_preferences(votes)
    disagreements = disagreement_examples(
        votes, dataset, pairwise
    )
    coverage = shared_subset_coverage(
        votes, dataset
    )

    outputs = {
        "agreement_summary.csv": summary,
        "model_preferences.csv": preferences,
        "disagreement_examples.csv": disagreements,
        "shared_subset_coverage.csv": coverage,
    }

    for filename, frame in outputs.items():
        path = OUTPUT_DIR / filename
        frame.to_csv(path, index=False)
        print(f"Saved: {path} ({len(frame)} rows)")

    print("\n=== DAY 16 HUMAN AGREEMENT ===")
    print("Participants:", votes["participant_id"].nunique())
    print("Votes:", len(votes))
    print("Unique sentences:", votes["evaluation_id"].nunique())
    print("Pairwise comparisons:", len(pairwise))

    print("\nAgreement summary:")
    print(summary.to_string(index=False))

    print("\nShared subset coverage:")
    print(
        coverage.groupby("domain")[
            "participant_count"
        ].agg(["min", "max", "mean"])
        .to_string()
    )


if __name__ == "__main__":
    main()
