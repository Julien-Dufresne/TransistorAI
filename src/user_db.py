
"""
TransistorAI — Persistent user management.

Stores user profiles independently from the vocabulary
and human preference databases.
"""

import sqlite3
import uuid

from datetime import datetime, timezone
from pathlib import Path


DB_PATH = (
    Path(__file__).resolve().parents[1]
    / "data/users/users.sqlite3"
)


def connect_users_db():
    """Open the user database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_users_db():
    """Create the user registry if it does not exist."""
    with connect_users_db() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL COLLATE NOCASE UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )


def create_user(username):
    """
    Create a user with a stable UUID.

    Usernames are unique, ignoring capitalization.
    """
    username = username.strip()

    if not username:
        raise ValueError("Username cannot be empty.")

    if len(username) > 50:
        raise ValueError("Username cannot exceed 50 characters.")

    initialize_users_db()

    user_id = str(uuid.uuid4())
    created_at = datetime.now(timezone.utc).isoformat()

    with connect_users_db() as connection:
        try:
            connection.execute(
                """
                INSERT INTO users (user_id, username, created_at)
                VALUES (?, ?, ?)
                """,
                (user_id, username, created_at),
            )
        except sqlite3.IntegrityError as exc:
            raise ValueError(
                f"Username '{username}' already exists."
            ) from exc

    return {
        "user_id": user_id,
        "username": username,
        "created_at": created_at,
    }


def get_users():
    """Return all registered users alphabetically."""
    initialize_users_db()

    with connect_users_db() as connection:
        rows = connection.execute(
            """
            SELECT user_id, username, created_at
            FROM users
            ORDER BY username COLLATE NOCASE
            """
        ).fetchall()

    return [
        {
            "user_id": row[0],
            "username": row[1],
            "created_at": row[2],
        }
        for row in rows
    ]


def get_user(user_id):
    """Return one user profile by its unique ID."""
    initialize_users_db()

    with connect_users_db() as connection:
        row = connection.execute(
            """
            SELECT user_id, username, created_at
            FROM users
            WHERE user_id = ?
            """,
            (user_id,),
        ).fetchone()

    if row is None:
        return None

    return {
        "user_id": row[0],
        "username": row[1],
        "created_at": row[2],
    }
