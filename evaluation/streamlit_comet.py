
from pathlib import Path

import streamlit as st
import torch

from comet import load_from_checkpoint


COMET_MODEL_ID = "Unbabel/wmt22-comet-da"

COMET_CHECKPOINT = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--Unbabel--wmt22-comet-da"
    / "snapshots"
    / "2760a223ac957f30acfb18c8aa649b01cf1d75f2"
    / "checkpoints/model.ckpt"
)


@st.cache_resource(show_spinner="Loading local COMET model...")
def load_comet_model():
    """
    Load the existing COMET checkpoint without calling
    Hugging Face download_model().
    """

    if not COMET_CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"COMET checkpoint not found: {COMET_CHECKPOINT}"
        )

    print("COMET DEBUG 1: Local checkpoint found", flush=True)

    model = load_from_checkpoint(
        str(COMET_CHECKPOINT)
    )

    print("COMET DEBUG 2: Checkpoint loaded", flush=True)

    model.eval()

    print("COMET DEBUG 3: Model ready", flush=True)

    return model


def calculate_comet_scores(
    source: str,
    user_translation: str,
    model_translation: str,
    reference: str,
) -> dict:
    """
    Compare two translations using the same COMET model.
    """

    comet_model = load_comet_model()

    samples = [
        {
            "src": source,
            "mt": user_translation,
            "ref": reference,
        },
        {
            "src": source,
            "mt": model_translation,
            "ref": reference,
        },
    ]

    prediction = comet_model.predict(
    samples,
    batch_size=2,
    gpus=0,
    num_workers=1,
    progress_bar=False,
)

    return {
        "user_comet": float(prediction.scores[0]),
        "model_comet": float(prediction.scores[1]),
    }
