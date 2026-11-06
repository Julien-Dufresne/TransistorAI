
"""Run the existing COMET evaluation in an isolated environment."""

import json
import sys

from evaluation.streamlit_comet import calculate_comet_scores


def main():
    payload = json.load(sys.stdin)

    scores = calculate_comet_scores(
        source=payload["source"],
        user_translation=payload["user_translation"],
        model_translation=payload["model_translation"],
        reference=payload["reference"],
    )

    print(json.dumps(scores), flush=True)


if __name__ == "__main__":
    main()
