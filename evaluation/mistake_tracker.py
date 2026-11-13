
"""Persistent, user-specific translation mistake tracking."""

import sqlite3
import uuid
from datetime import datetime, timezone

from evaluation.streamlit_attempts import connect, initialize_database


CATEGORIES = [
    "grammar", "conjugation", "plurals", "capitalization",
    "omissions", "word_order", "articles", "prepositions",
    "vocabulary", "spelling", "meaning", "terminology",
    "numbers", "placeholders", "formatting", "other",
]


def initialize_mistake_database():
    initialize_database()

    with connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS mistakes (
                mistake_id TEXT PRIMARY KEY,
                attempt_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                category TEXT NOT NULL,
                original_text TEXT NOT NULL,
                corrected_text TEXT NOT NULL,
                explanation TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'new',
                created_at TEXT NOT NULL,
                reviewed_at TEXT,
                FOREIGN KEY (attempt_id)
                    REFERENCES attempts(attempt_id)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_mistakes_user
            ON mistakes(user_id, created_at)
        """)


def save_mistakes(attempt_id, user_id, mistakes):
    """Save confirmed mistakes for an owned attempt."""
    if not user_id:
        raise ValueError("user_id is required")

    initialize_mistake_database()

    saved_ids = []

    with connect() as conn:
        attempt = conn.execute("""
            SELECT attempt_id FROM attempts
            WHERE attempt_id = ? AND user_id = ?
        """, (attempt_id, user_id)).fetchone()

        if attempt is None:
            raise ValueError("Attempt not found for this user")

        for mistake in mistakes:
            category = mistake["category"]

            if category not in CATEGORIES:
                raise ValueError(f"Unknown category: {category}")

            mistake_id = str(uuid.uuid4())

            conn.execute("""
                INSERT INTO mistakes (
                    mistake_id, attempt_id, user_id,
                    category, original_text, corrected_text,
                    explanation, status, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'new', ?)
            """, (
                mistake_id,
                attempt_id,
                user_id,
                category,
                mistake.get("original_text", ""),
                mistake.get("corrected_text", ""),
                mistake.get("explanation", ""),
                datetime.now(timezone.utc).isoformat(),
            ))

            saved_ids.append(mistake_id)

    return saved_ids


def get_user_mistakes(user_id):
    """Retrieve all mistakes belonging to a learner."""
    if not user_id:
        raise ValueError("user_id is required")

    initialize_mistake_database()

    with connect() as conn:
        conn.row_factory = sqlite3.Row

        rows = conn.execute("""
            SELECT *
            FROM mistakes
            WHERE user_id = ?
            ORDER BY created_at DESC
        """, (user_id,)).fetchall()

    return [dict(row) for row in rows]


def update_mistake_status(mistake_id, user_id, status):
    """Update review status without affecting other users."""
    if status not in ("new", "reviewing", "mastered"):
        raise ValueError("Invalid mistake status")

    initialize_mistake_database()

    with connect() as conn:
        cursor = conn.execute("""
            UPDATE mistakes
            SET status = ?, reviewed_at = ?
            WHERE mistake_id = ? AND user_id = ?
        """, (
            status,
            datetime.now(timezone.utc).isoformat(),
            mistake_id,
            user_id,
        ))

        if cursor.rowcount == 0:
            raise ValueError("Mistake not found for this user")
