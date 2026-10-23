
from pathlib import Path
from datetime import datetime, timezone
import csv

# Resolve paths relative to the project, not the terminal.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# Create the evaluation folder if it does not exist.
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)


def append_csv(filename, record):
    """Append a dictionary as one row in a CSV file."""

    file_path = EVALUATION_DIR / filename

    # Write column headers only when creating a new file.
    new_file = not file_path.exists() or file_path.stat().st_size == 0

    with file_path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(record.keys()),
        )

        if new_file:
            writer.writeheader()

        writer.writerow(record)


def log_translation(source, translation, model_id, language_pair, latency_ms):
    """Save a translation experiment with its metadata."""

    append_csv("translation_log.csv", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "language_pair": language_pair,
        "source": source,
        "translation": translation,
        "model_id": model_id,
        "latency_ms": round(latency_ms, 2),
    })
