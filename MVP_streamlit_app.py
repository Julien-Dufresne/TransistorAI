# STEP 1 — Import libraries
# Streamlit creates the web interface.
# PyTorch runs the neural network.
# Transformers loads the pretrained model and tokenizer.
# time.perf_counter() measures elapsed time precisely.

import time
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Import the five exercises and comparison function.
from src.learning import EXERCISES, compare_translations

# Import CSV logging functions.
from src.translation_logger import log_translation, append_csv

# Used to timestamp learning attempts.
from datetime import datetime, timezone

import pandas as pd

from evaluation.streamlit_metrics import (
    calculate_translation_metrics,
)
from evaluation.streamlit_comet import calculate_comet_scores

# STEP 2 — Define model and decoding configuration
# Keep the same configuration as Day 2 to make results
# comparable between the notebook and Streamlit demo.

# Rather than manually changing the model ID every time, you can create a dictionary:
MODELS = {
    "en-de": "Helsinki-NLP/opus-mt-en-de",
    "de-en": "Helsinki-NLP/opus-mt-de-en",
}

# =======> Change LANGUAGE_PAIR to "en-de" whenever you want English → German. Remember to choose the corresponding test sentences and direction-specific output filenames.
LANGUAGE_PAIR = "de-en"

MODEL_ID = MODELS[LANGUAGE_PAIR]

DECODING_CONFIG = {
    "num_beams": 4,          # Explore four candidate sequences
    "do_sample": False,     # Use deterministic beam search
    "max_new_tokens": 128,  # Limit generated output length
}


# STEP 3 — Select the computation device
# CUDA is used for NVIDIA GPUs.
# MPS enables Apple Silicon GPU acceleration.
# CPU is the fallback when neither is available.

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# STEP 4 — Load and cache the pretrained model
# Streamlit reruns the script when the user interacts with
# widgets. Without caching, the model could reload each time.
#
# @st.cache_resource keeps the model and tokenizer in memory.
# This reduces overhead for subsequent translations.

# @st.cache_resource
def load_translation_model(model_id):

    print("MODEL DEBUG 1: Loading tokenizer", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=True,
    )

    print("MODEL DEBUG 2: Tokenizer loaded", flush=True)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        local_files_only=True,
    )

    print("MODEL DEBUG 3: Model weights loaded", flush=True)

    model = model.to(DEVICE)

    print(f"MODEL DEBUG 4: Model moved to {DEVICE}", flush=True)

    model.eval()

    print("MODEL DEBUG 5: Model ready", flush=True)

    return tokenizer, model


# STEP 5 — Define the translation function
# This function accepts German text and returns:
# 1. The English translation
# 2. The measured inference latency in milliseconds

def translate(text, tokenizer, model):

    # Convert input text into numerical tokens.
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    # Move input tensors to the same device as the model.
    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # Synchronize GPU work before timing.
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()

    start_time = time.perf_counter()

    # Inference mode disables gradient computation.
    # This reduces memory use and computational overhead.
    with torch.inference_mode():
        output_tokens = model.generate(
            **inputs,
            **DECODING_CONFIG,
        )

    # GPU operations can execute asynchronously.
    # Synchronization ensures generation has finished.
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()

    elapsed_ms = (
        time.perf_counter() - start_time
    ) * 1000

    # Convert generated token IDs into readable English.
    translation = tokenizer.decode(
        output_tokens[0],
        skip_special_tokens=True,
    )

    return translation, elapsed_ms




# STEP 6 — Configure the Streamlit interface
# The page title and layout define the user experience.

st.set_page_config(
    page_title="TransistorAI",
    page_icon="🌍",
    layout="centered",
)

st.title("🌍 TransistorAI")
# STEP 6A — Create two independent interface tabs.
# The first tab handles normal translation.
# The second tab contains interactive translation exercises.

translate_tab, practice_tab = st.tabs([
    "🌍 Translate",
    "🎓 Test Yourself",
])


st.subheader("German → English Translation")

st.write(
    "Translate German text using a pretrained "
    "Transformer-based neural machine translation model."
)



# ============================================================
# TAB 1 — TRANSLATE
# ============================================================

with translate_tab:

    st.subheader("German → English Translation")

    source_text = st.text_area(
        "Enter German text:",
        placeholder="Enter your German text here...",
        height=120,
    )

    if st.button("Translate", type="primary"):

        if not source_text.strip():
            st.warning("Please enter some German text.")

        else:
            with st.spinner("Generating translation..."):

                print("DEBUG 1: Loading model", flush=True)

                tokenizer, model = load_translation_model(
                    MODELS["de-en"]
                )

                print("DEBUG 2: Model loaded", flush=True)

                translation, latency_ms = translate(
                    source_text,
                    tokenizer,
                    model,
                )

                print("DEBUG 3: Translation completed", flush=True)

            log_translation(
                source=source_text,
                translation=translation,
                model_id=MODEL_ID,
                language_pair=LANGUAGE_PAIR,
                latency_ms=latency_ms,
            )

            st.success("Translation completed!")

            st.markdown("### German source")
            st.write(source_text)

            st.markdown("### English translation")
            st.write(translation)

            st.markdown("### Model information")
            st.write(f"**Model:** {MODEL_ID}")
            st.write(f"**Device:** {DEVICE}")
            st.write(f"**Latency:** {latency_ms:.2f} ms")

            with st.expander("Decoding configuration"):
                st.json(DECODING_CONFIG)


# ============================================================
# TAB 2 — TEST YOURSELF SECTION
# ============================================================


with practice_tab:

    st.subheader("Test Yourself — German → English")

    @st.cache_data
    def load_exercises():
        df = pd.read_csv(
            "data/exercises/translation_exercises.csv"
        )

        required = {
            "id", "domain", "source", "reference", "focus"
        }
        if not required.issubset(df.columns):
            raise ValueError("Missing exercise CSV columns.")

        return df

    exercises = load_exercises()

    domain = st.radio(
        "Choose exercise category",
        ["general", "technical"],
        horizontal=True,
        format_func=lambda x: x.title(),
    )

    filtered = exercises[
        exercises["domain"] == domain
    ].reset_index(drop=True)

    exercise_index = st.selectbox(
        "Select an exercise",
        range(len(filtered)),
        format_func=lambda i: (
            f"Exercise {i + 1} — {filtered.iloc[i]['focus']}"
        ),
    )

    exercise = filtered.iloc[exercise_index]

    st.info(exercise["source"])
    st.caption(f"Focus: {exercise['focus']}")

    user_answer = st.text_area(
        "Your English translation:",
        key=f"answer_{domain}_{exercise['id']}",
        height=100,
    )

    enable_comet = st.checkbox(
    "Calculate COMET",
    value=True,
    help="Evaluate translation quality using a neural model.",
)
    
    
    if st.button("Evaluate translation", type="primary"):

        if not user_answer.strip():
            st.warning("Please enter a translation.")

        else:
            # Retrieve the independent English reference.
            reference = str(exercise["reference"])

            with st.spinner("Generating model translation..."):

                print("PRACTICE DEBUG 1: Loading model", flush=True)

                tokenizer, model = load_translation_model(
                    MODELS["de-en"]
                )

                print("PRACTICE DEBUG 2: Model loaded", flush=True)

                model_answer, latency_ms = translate(
                    exercise["source"],
                    tokenizer,
                    model,
                )

                print("PRACTICE DEBUG 3: Translation completed", flush=True)
                print(f"Model output: {model_answer}", flush=True)

            # Evaluate both translations against the reference.
            user_scores = calculate_translation_metrics(
                user_answer,
                reference,
            )

            model_scores = calculate_translation_metrics(
                model_answer,
                reference,
            )

            # Optional neural evaluation.
            comet_scores = None

            if enable_comet:
                try:
                    with st.spinner("Calculating COMET scores..."):
                        comet_scores = calculate_comet_scores(
                            source=exercise["source"],
                            user_translation=user_answer,
                            model_translation=model_answer,
                            reference=reference,
                        )
                except Exception as error:
                    st.warning(
                        f"COMET evaluation unavailable: {error}"
                    )

            # Display translations.
            st.markdown("### Translation comparison")
            st.write("**German source:**", exercise["source"])
            st.write("**English reference:**", reference)
            st.write("**Your translation:**", user_answer)
            st.write("**Model translation:**", model_answer)

            # Build the BLEU and chrF results.
            results = pd.DataFrame({
                "Metric": ["BLEU", "chrF"],
                "Your translation": [
                    user_scores["bleu"],
                    user_scores["chrf"],
                ],
                "Model translation": [
                    model_scores["bleu"],
                    model_scores["chrf"],
                ],
            })

            # Include COMET if evaluation succeeded.
            if comet_scores is not None:
                results = pd.concat([
                    results,
                    pd.DataFrame([{
                        "Metric": "COMET",
                        "Your translation": round(
                            comet_scores["user_comet"], 4
                        ),
                        "Model translation": round(
                            comet_scores["model_comet"], 4
                        ),
                    }])
                ], ignore_index=True)

            st.markdown("### Evaluation metrics")
            st.dataframe(results, hide_index=True)

            # BLEU and chrF: 0–100 scale.
            st.markdown("#### BLEU and chrF")
            st.bar_chart(
                results[
                    results["Metric"].isin(["BLEU", "chrF"])
                ].set_index("Metric")
            )

            # COMET: separate scoring scale.
            if comet_scores is not None:
                st.markdown("#### COMET")
                st.bar_chart(
                    pd.DataFrame({
                        "Translation": [
                            "Your translation",
                            "Model translation",
                        ],
                        "COMET": [
                            comet_scores["user_comet"],
                            comet_scores["model_comet"],
                        ],
                    }).set_index("Translation")
                )

            st.caption(
                "BLEU and chrF measure similarity to the "
                "reference. COMET estimates translation "
                "quality using a neural evaluation model. "
                "Higher scores generally indicate better "
                "translations, but none guarantees correctness."
            )

            st.write(
                f"Model inference latency: {latency_ms:.2f} ms"
            )

