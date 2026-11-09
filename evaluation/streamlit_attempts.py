
"""Persistent storage for Test Yourself attempts.

Supports multiple users while preserving historical attempts.
"""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path


DB_PATH = Path("data/exercises/translation_attempts.sqlite3")


def connect():
    """Connect to the translation attempts database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database():
    """Initialize the database and migrate older schemas safely."""

    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                exercise_id TEXT NOT NULL,
                domain TEXT NOT NULL,
                source TEXT NOT NULL,
                reference TEXT NOT NULL,
                user_translation TEXT NOT NULL,
                error_tags TEXT NOT NULL,
                rule_feedback TEXT NOT NULL,
                llm_feedback TEXT,
                model_translation TEXT,
                bleu REAL,
                chrf REAL,
                comet REAL,
                model_bleu REAL,
                model_chrf REAL,
                model_comet REAL,
                scores_revealed_at TEXT,
                user_id TEXT
            )
        """)

        # Migrate databases created before multi-user support.
        columns = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(attempts)"
            )
        }

        if "user_id" not in columns:
            conn.execute(
                "ALTER TABLE attempts ADD COLUMN user_id TEXT"
            )

        # Improve performance when querying user attempts.
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_attempts_user_id
            ON attempts(user_id)
        """)


def save_attempt(
    exercise_id,
    domain,
    source,
    reference,
    user_translation,
    analysis,
    user_id,
):
    """Save a translation attempt belonging to a specific user."""

    if not user_id:
        raise ValueError("user_id is required.")

    initialize_database()

    attempt_id = str(uuid.uuid4())

    tags = [
        item["category"]
        for item in analysis
        if item["status"] == "review_suggested"
    ]

    with connect() as conn:
        conn.execute("""
            INSERT INTO attempts (
                attempt_id,
                created_at,
                exercise_id,
                domain,
                source,
                reference,
                user_translation,
                error_tags,
                rule_feedback,
                user_id
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            attempt_id,
            datetime.now(timezone.utc).isoformat(),
            str(exercise_id),
            domain,
            source,
            reference,
            user_translation,
            json.dumps(tags),
            json.dumps(analysis),
            user_id,
        ))

    return attempt_id


def get_attempt(attempt_id, user_id):
    """Retrieve an attempt only if it belongs to the user."""

    if not user_id:
        raise ValueError("user_id is required.")

    initialize_database()

    with connect() as conn:
        conn.row_factory = sqlite3.Row

        row = conn.execute("""
            SELECT *
            FROM attempts
            WHERE attempt_id = ?
              AND user_id = ?
        """, (
            attempt_id,
            user_id,
        )).fetchone()

    return dict(row) if row else None


def save_scores(
    attempt_id,
    user_scores,
    model_scores,
    model_translation,
    user_id,
    comet_scores=None,
):
    """Save evaluation scores only for the attempt's owner."""

    if not user_id:
        raise ValueError("user_id is required.")

    initialize_database()

    with connect() as conn:
        cursor = conn.execute("""
            UPDATE attempts
            SET
                model_translation = ?,
                bleu = ?,
                chrf = ?,
                comet = ?,
                model_bleu = ?,
                model_chrf = ?,
                model_comet = ?,
                scores_revealed_at = ?
            WHERE attempt_id = ?
              AND user_id = ?
        """, (
            model_translation,
            user_scores["bleu"],
            user_scores["chrf"],
            (
                comet_scores["user_comet"]
                if comet_scores else None
            ),
            model_scores["bleu"],
            model_scores["chrf"],
            (
                comet_scores["model_comet"]
                if comet_scores else None
            ),
            datetime.now(timezone.utc).isoformat(),
            attempt_id,
            user_id,
        ))

        if cursor.rowcount == 0:
            raise ValueError(
                "Attempt not found or does not belong "
                "to the selected user."
            )


def get_user_attempts(user_id, limit=100):
    """Return a user's attempts, newest first."""

    if not user_id:
        raise ValueError("user_id is required.")

    initialize_database()

    with connect() as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT *
            FROM attempts
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
        """, (
            user_id,
            limit,
        )).fetchall()

    return [dict(row) for row in rows]
