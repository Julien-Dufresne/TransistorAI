
"""
Day 18 — Human evaluation analytics validation.

Tests use synthetic participant votes and do not access
or modify the production SQLite database.
"""

import pandas as pd
import pytest

from evaluation.day16_analyze_human_agreement import (
    calculate_pairwise_agreement,
    agreement_summary,
    model_preferences,
)


@pytest.fixture
def sample_dataset():
    """Create a small dataset containing two evaluation domains."""
    return pd.DataFrame(
        [
            {"evaluation_id": "G001", "domain": "general"},
            {"evaluation_id": "T001", "domain": "technical"},
            {"evaluation_id": "T002", "domain": "technical"},
        ]
    ).set_index("evaluation_id", drop=False)


@pytest.fixture
def sample_votes():
    """Create votes with agreement, disagreement and grouped models."""
    return pd.DataFrame(
        [
            {
                "participant_id": "P1",
                "evaluation_id": "G001",
                "domain": "general",
                "selected_text": "Hello.",
                "models": ["E0"],
            },
            {
                "participant_id": "P2",
                "evaluation_id": "G001",
                "domain": "general",
                "selected_text": "Hello.",
                "models": ["E0"],
            },
            {
                "participant_id": "P1",
                "evaluation_id": "T001",
                "domain": "technical",
                "selected_text": "Restart the server.",
                "models": ["E1", "E2"],
            },
            {
                "participant_id": "P2",
                "evaluation_id": "T001",
                "domain": "technical",
                "selected_text": "Reboot the server.",
                "models": ["E3"],
            },
            {
                "participant_id": "P1",
                "evaluation_id": "T002",
                "domain": "technical",
                "selected_text": "Save the file.",
                "models": ["E1"],
            },
        ]
    )


def test_fractional_preferences_sum_to_votes(sample_votes):
    """Each vote must contribute exactly one unit of fractional credit."""
    preferences = model_preferences(sample_votes)

    total = preferences["fractional_selections"].sum()

    assert total == pytest.approx(len(sample_votes))


def test_grouped_models_split_credit(sample_votes):
    """Identical translations split credit equally between models."""
    preferences = model_preferences(sample_votes)

    participant = preferences[
        (preferences["participant_id"] == "P1")
        & (preferences["domain"] == "technical")
    ]

    credits = participant.set_index("model")[
        "fractional_selections"
    ]

    # T001 contributes 0.5 to E1 and 0.5 to E2.
    # T002 contributes another 1.0 to E1.
    assert credits["E1"] == pytest.approx(1.5)
    assert credits["E2"] == pytest.approx(0.5)


def test_pairwise_agreement_uses_shared_sentences(
    sample_votes,
    sample_dataset,
):
    """Only sentences evaluated by multiple participants form pairs."""
    pairwise = calculate_pairwise_agreement(
        sample_votes,
        sample_dataset,
    )

    assert len(pairwise) == 2
    assert set(pairwise["evaluation_id"]) == {"G001", "T001"}


def test_agreement_by_domain(sample_votes, sample_dataset):
    """Verify agreement and disagreement are separated by domain."""
    pairwise = calculate_pairwise_agreement(
        sample_votes,
        sample_dataset,
    )

    summary = agreement_summary(pairwise).set_index("domain")

    assert summary.loc["general", "exact_agreement_percent"] == 100.0
    assert summary.loc["technical", "exact_agreement_percent"] == 0.0
    assert summary.loc["overall", "exact_agreement_percent"] == 50.0


def test_empty_pairwise_agreement():
    """No overlapping votes should produce an empty agreement summary."""
    empty = pd.DataFrame(
        columns=[
            "evaluation_id",
            "domain",
            "participant_a",
            "participant_b",
            "exact_agreement",
            "model_overlap",
        ]
    )

    summary = agreement_summary(empty)

    assert len(summary) == 3
    assert summary["participant_pairs"].eq(0).all()
    assert summary["exact_agreement_percent"].isna().all()
