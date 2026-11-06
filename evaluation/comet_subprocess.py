
"""Call COMET using the isolated Python environment."""

import json
import subprocess
from pathlib import Path

COMET_PYTHON = Path(
    "/opt/anaconda3/envs/transistorai-comet-test/bin/python"
)


def calculate_comet_isolated(
    source: str,
    user_translation: str,
    model_translation: str,
    reference: str,
) -> dict:

    if not COMET_PYTHON.is_file():
        raise FileNotFoundError(
            f"COMET Python environment not found: {COMET_PYTHON}"
        )

    payload = {
        "source": source,
        "user_translation": user_translation,
        "model_translation": model_translation,
        "reference": reference,
    }

    result = subprocess.run(
        [
            str(COMET_PYTHON),
            "-m",
            "evaluation.comet_worker",
        ],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
        cwd=Path(__file__).resolve().parent.parent,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "COMET evaluation failed:\n"
            + result.stderr[-3000:]
        )

    # The existing loader prints debug messages to stdout.
    for line in reversed(result.stdout.splitlines()):
        try:
            scores = json.loads(line)
        except json.JSONDecodeError:
            continue

        if (
            isinstance(scores, dict)
            and "user_comet" in scores
            and "model_comet" in scores
        ):
            return {
                "user_comet": float(scores["user_comet"]),
                "model_comet": float(scores["model_comet"]),
            }

    raise RuntimeError(
        "COMET worker returned no valid scores."
    )
