
"""
Day 19 — SQLite storage for the Daily German Vocabulary Trainer.

Stores vocabulary words, daily assignments, learner progress,
and quiz attempts independently from human evaluation data.
"""

import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/vocabulary/vocabulary.sqlite3"
)


def connect_db():
    """Open the vocabulary database with foreign keys enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_database():
    """Create the vocabulary tables if they do not already exist."""

    with connect_db() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS vocabulary (
                word_id INTEGER PRIMARY KEY AUTOINCREMENT,
                german TEXT NOT NULL,
                english TEXT NOT NULL,
                level TEXT NOT NULL DEFAULT 'A1',
                category TEXT NOT NULL DEFAULT 'general',
                UNIQUE(german, english)
            );

            CREATE TABLE IF NOT EXISTS learner_progress (
                learner_id TEXT NOT NULL,
                word_id INTEGER NOT NULL,
                first_seen TEXT NOT NULL,
                last_reviewed TEXT,
                correct_count INTEGER NOT NULL DEFAULT 0,
                incorrect_count INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (learner_id, word_id),
                FOREIGN KEY (word_id) REFERENCES vocabulary(word_id)
            );

            CREATE TABLE IF NOT EXISTS daily_assignments (
                learner_id TEXT NOT NULL,
                study_date TEXT NOT NULL,
                word_id INTEGER NOT NULL,
                assignment_type TEXT NOT NULL
                    CHECK (assignment_type IN ('new', 'review')),
                PRIMARY KEY (learner_id, study_date, word_id),
                FOREIGN KEY (word_id) REFERENCES vocabulary(word_id)
            );

            CREATE TABLE IF NOT EXISTS quiz_attempts (
                attempt_id INTEGER PRIMARY KEY AUTOINCREMENT,
                learner_id TEXT NOT NULL,
                word_id INTEGER NOT NULL,
                study_date TEXT NOT NULL,
                user_answer TEXT NOT NULL,
                is_correct INTEGER NOT NULL
                    CHECK (is_correct IN (0, 1)),
                attempted_at TEXT NOT NULL,
                FOREIGN KEY (word_id) REFERENCES vocabulary(word_id)
            );
            """
        )

        connection.commit()


def add_vocabulary(german, english, level="A1", category="general"):
    """Insert a vocabulary entry without duplicating existing words."""

    with connect_db() as connection:
        connection.execute(
            """
            INSERT OR IGNORE INTO vocabulary
                (german, english, level, category)
            VALUES (?, ?, ?, ?)
            """,
            (german, english, level, category),
        )
        connection.commit()


def record_attempt(learner_id, word_id, user_answer, is_correct):
    """Save a quiz answer and update the learner's word statistics."""

    today = date.today().isoformat()
    timestamp = datetime.now(timezone.utc).isoformat()

    with connect_db() as connection:
        # Ensure the learner has a progress record for this word.
        connection.execute(
            """
            INSERT OR IGNORE INTO learner_progress
                (learner_id, word_id, first_seen)
            VALUES (?, ?, ?)
            """,
            (learner_id, word_id, today),
        )

        # Record the individual quiz attempt.
        connection.execute(
            """
            INSERT INTO quiz_attempts
                (learner_id, word_id, study_date,
                 user_answer, is_correct, attempted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                learner_id,
                word_id,
                today,
                user_answer,
                int(is_correct),
                timestamp,
            ),
        )

        # Update the learner's cumulative statistics.
        connection.execute(
            """
            UPDATE learner_progress
            SET
                last_reviewed = ?,
                correct_count = correct_count + ?,
                incorrect_count = incorrect_count + ?
            WHERE learner_id = ? AND word_id = ?
            """,
            (
                today,
                int(is_correct),
                int(not is_correct),
                learner_id,
                word_id,
            ),
        )

        connection.commit()


def submit_daily_answer(learner_id, word_id, user_answer, is_correct,
                        study_date=None):
    """
    Save one answer per learner, word and study date.

    Returns True if saved, False if the word was already answered.
    """
    today = study_date or date.today().isoformat()
    timestamp = datetime.now(timezone.utc).isoformat()

    with connect_db() as connection:
        # Lock the database for an atomic check-and-insert.
        connection.execute("BEGIN IMMEDIATE")

        assigned = connection.execute(
            """
            SELECT 1
            FROM daily_assignments
            WHERE learner_id = ?
              AND study_date = ?
              AND word_id = ?
            """,
            (learner_id, today, word_id),
        ).fetchone()

        if assigned is None:
            raise ValueError("Word is not assigned to this learner today.")

        existing = connection.execute(
            """
            SELECT 1
            FROM quiz_attempts
            WHERE learner_id = ?
              AND study_date = ?
              AND word_id = ?
            """,
            (learner_id, today, word_id),
        ).fetchone()

        if existing:
            return False

        connection.execute(
            """
            INSERT INTO quiz_attempts
                (learner_id, word_id, study_date,
                 user_answer, is_correct, attempted_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                learner_id, word_id, today,
                user_answer, int(is_correct), timestamp,
            ),
        )

        connection.execute(
            """
            INSERT OR IGNORE INTO learner_progress
                (learner_id, word_id, first_seen)
            VALUES (?, ?, ?)
            """,
            (learner_id, word_id, today),
        )

        connection.execute(
            """
            UPDATE learner_progress
            SET last_reviewed = ?,
                correct_count = correct_count + ?,
                incorrect_count = incorrect_count + ?
            WHERE learner_id = ? AND word_id = ?
            """,
            (
                today,
                int(is_correct),
                int(not is_correct),
                learner_id,
                word_id,
            ),
        )

    return True


def get_daily_attempts(learner_id, study_date=None):
    """Return today's submitted quiz answers, indexed by word ID."""
    today = study_date or date.today().isoformat()

    with connect_db() as connection:
        rows = connection.execute(
            """
            SELECT word_id, user_answer, is_correct
            FROM quiz_attempts
            WHERE learner_id = ?
              AND study_date = ?
            """,
            (learner_id, today),
        ).fetchall()

    return {
        word_id: {
            "answer": answer,
            "correct": bool(is_correct),
        }
        for word_id, answer, is_correct in rows
    }
