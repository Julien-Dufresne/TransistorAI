
"""
Import vocabulary entries from CSV into the SQLite database.

The database uses a unique German-English constraint,
so running this importer repeatedly will not duplicate entries.
"""

import csv
from pathlib import Path

from src.vocabulary_db import (
    initialize_database,
    add_vocabulary,
)

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/vocabulary/b1_b2_vocabulary.csv"


def main():
    """Initialize the database and import the vocabulary bank."""
    initialize_database()

    with CSV_PATH.open(
        "r", encoding="utf-8", newline=""
    ) as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    for row in rows:
        add_vocabulary(
            german=row["german"],
            english=row["english"],
            level=row["level"],
            category=row["category"],
        )

    print(f"Processed {len(rows)} vocabulary entries.")


if __name__ == "__main__":
    main()
