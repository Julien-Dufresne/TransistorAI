import hashlib
import json
import random
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATASET_PATH = Path("data/human_evaluation/preference_dataset.csv")
DB_PATH = Path("data/human_evaluation/preference_votes.sqlite3")

MODELS = ["E0", "E1", "E2", "E3"]


def connect_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH, timeout=30)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS votes (
            participant_id TEXT NOT NULL,
            evaluation_id TEXT NOT NULL,
            domain TEXT NOT NULL,
            selected_option TEXT NOT NULL,
            selected_text TEXT NOT NULL,
            selected_models TEXT NOT NULL,
            option_mapping TEXT NOT NULL,
            voted_at TEXT NOT NULL,
            PRIMARY KEY (participant_id, evaluation_id)
        )
    """)

    conn.commit()
    return conn


def load_dataset():
    df = pd.read_csv(DATASET_PATH, keep_default_na=False)

    expected = {
        "evaluation_id", "domain", "source", "reference",
        "translation_E0", "translation_E1",
        "translation_E2", "translation_E3",
    }

    if not expected.issubset(df.columns):
        raise ValueError("Preference dataset has missing columns")

    if df["evaluation_id"].duplicated().any():
        raise ValueError("Duplicate evaluation IDs")

    if len(df) != 200:
        raise ValueError(f"Expected 200 sentences, got {len(df)}")

    return df


def deterministic_seed(participant_id, evaluation_id):
    value = f"{participant_id}:{evaluation_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def build_options(row, participant_id):
    # Group models that generated exactly the same translation.
    grouped = {}

    for model in MODELS:
        translation = str(row[f"translation_{model}"]).strip()
        grouped.setdefault(translation, []).append(model)

    options = [
        {"text": text, "models": models}
        for text, models in grouped.items()
    ]

    rng = random.Random(
        deterministic_seed(participant_id, row["evaluation_id"])
    )
    rng.shuffle(options)

    for index, option in enumerate(options):
        option["label"] = chr(ord("A") + index)

    return options


def get_completed_ids(participant_id):
    with connect_db() as conn:
        rows = conn.execute(
            "SELECT evaluation_id FROM votes WHERE participant_id = ?",
            (participant_id,),
        ).fetchall()

    return {row[0] for row in rows}


def save_vote(participant_id, row, selected_label, options):
    matching = [
        option for option in options
        if option["label"] == selected_label
    ]

    if len(matching) != 1:
        raise ValueError("Invalid selected option")

    selected = matching[0]

    mapping = {
        option["label"]: {
            "text": option["text"],
            "models": option["models"],
        }
        for option in options
    }

    with connect_db() as conn:
        conn.execute(
            """
            INSERT INTO votes (
                participant_id,
                evaluation_id,
                domain,
                selected_option,
                selected_text,
                selected_models,
                option_mapping,
                voted_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                participant_id,
                row["evaluation_id"],
                row["domain"],
                selected_label,
                selected["text"],
                json.dumps(selected["models"]),
                json.dumps(mapping, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()


def export_votes():
    with connect_db() as conn:
        return pd.read_sql_query(
            "SELECT * FROM votes ORDER BY voted_at",
            conn,
        )

def participant_exists(participant_id):
    with connect_db() as conn:
        row = conn.execute(
            """
            SELECT 1
            FROM votes
            WHERE participant_id = ?
            LIMIT 1
            """,
            (participant_id,),
        ).fetchone()

    return row is not None