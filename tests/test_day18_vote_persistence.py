
"""
Day 18 — SQLite vote persistence and duplicate protection tests.

All tests use a temporary database created by pytest.
The production human preference database is never modified.
"""

import json
import sqlite3

import pytest

from evaluation import day15_human_preference as preference


@pytest.fixture
def temporary_database(tmp_path, monkeypatch):
    """
    Redirect all database operations to an isolated SQLite file.

    monkeypatch automatically restores the original DB_PATH after
    each test, preventing accidental changes to real evaluation data.
    """
    database_path = tmp_path / "test_preference_votes.sqlite3"

    monkeypatch.setattr(
        preference,
        "DB_PATH",
        database_path,
    )

    return database_path


@pytest.fixture
def sample_vote():
    """Create a representative evaluation item and model options."""

    row = {
        "evaluation_id": "DAY18_TEST_001",
        "domain": "technical",
    }

    options = [
        {
            "label": "A",
            "text": "Restart the server.",
            "models": ["E0"],
        },
        {
            "label": "B",
            "text": "Reboot the server.",
            "models": ["E1", "E2"],
        },
        {
            "label": "C",
            "text": "Start the server again.",
            "models": ["E3"],
        },
    ]

    return row, options


def test_vote_persists_after_reconnection(
    temporary_database,
    sample_vote,
):
    """Verify a saved vote remains available in a new connection."""

    row, options = sample_vote

    preference.save_vote(
        "participant_001",
        row,
        "A",
        options,
    )

    # Open a separate connection to simulate a new application session.
    with sqlite3.connect(temporary_database) as connection:
        saved = connection.execute(
            """
            SELECT selected_option
            FROM votes
            WHERE participant_id = ?
            """,
            ("participant_001",),
        ).fetchone()

    assert saved is not None
    assert saved[0] == "A"


def test_duplicate_vote_is_rejected(
    temporary_database,
    sample_vote,
):
    """Verify that the same participant cannot vote twice."""

    row, options = sample_vote

    preference.save_vote(
        "participant_001", row, "A", options
    )

    # The composite primary key must reject a second vote.
    with pytest.raises(sqlite3.IntegrityError):
        preference.save_vote(
            "participant_001", row, "B", options
        )

    # Verify the original vote remains unchanged.
    with sqlite3.connect(temporary_database) as connection:
        saved = connection.execute(
            """
            SELECT selected_option
            FROM votes
            WHERE participant_id = ?
              AND evaluation_id = ?
            """,
            ("participant_001", "DAY18_TEST_001"),
        ).fetchone()

    assert saved[0] == "A"


def test_different_participants_can_vote(
    temporary_database,
    sample_vote,
):
    """Verify multiple participants can evaluate the same sentence."""

    row, options = sample_vote

    preference.save_vote(
        "participant_001", row, "A", options
    )

    preference.save_vote(
        "participant_002", row, "B", options
    )

    with sqlite3.connect(temporary_database) as connection:
        count = connection.execute(
            "SELECT COUNT(*) FROM votes"
        ).fetchone()[0]

    assert count == 2


def test_completed_ids_are_restored(
    temporary_database,
    sample_vote,
):
    """Verify completed evaluation IDs can be recovered."""

    row, options = sample_vote

    preference.save_vote(
        "participant_001", row, "A", options
    )

    completed = preference.get_completed_ids(
        "participant_001"
    )

    assert completed == {"DAY18_TEST_001"}


def test_participant_lookup(
    temporary_database,
    sample_vote,
):
    """Verify existing and unknown participants are distinguished."""

    row, options = sample_vote

    preference.save_vote(
        "participant_001", row, "A", options
    )

    assert preference.participant_exists("participant_001")
    assert not preference.participant_exists("unknown")


def test_tied_model_mapping_is_preserved(
    temporary_database,
    sample_vote,
):
    """Verify that identical translations retain all model IDs."""

    row, options = sample_vote

    preference.save_vote(
        "participant_001", row, "B", options
    )

    with sqlite3.connect(temporary_database) as connection:
        saved_models = connection.execute(
            """
            SELECT selected_models
            FROM votes
            WHERE participant_id = ?
            """,
            ("participant_001",),
        ).fetchone()[0]

    assert json.loads(saved_models) == ["E1", "E2"]
