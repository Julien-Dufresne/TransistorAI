
"""Persistent storage for Test Yourself attempts."""

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path("data/exercises/translation_attempts.sqlite3")


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)


def initialize_database():
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
                scores_revealed_at TEXT
            )
        """)


def save_attempt(
    exercise_id,
    domain,
    source,
    reference,
    user_translation,
    analysis,
):
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
                attempt_id, created_at, exercise_id,
                domain, source, reference,
                user_translation, error_tags, rule_feedback
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ))

    return attempt_id


def get_attempt(attempt_id):
    with connect() as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM attempts WHERE attempt_id = ?",
            (attempt_id,),
        ).fetchone()

    return dict(row) if row else None


def save_scores(
    attempt_id,
    user_scores,
    model_scores,
    model_translation,
    comet_scores=None,
):
    with connect() as conn:
        conn.execute("""
            UPDATE attempts SET
                model_translation = ?,
                bleu = ?,
                chrf = ?,
                comet = ?,
                model_bleu = ?,
                model_chrf = ?,
                model_comet = ?,
                scores_revealed_at = ?
            WHERE attempt_id = ?
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
        ))
