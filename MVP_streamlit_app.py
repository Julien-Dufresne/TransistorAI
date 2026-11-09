# STEP 1 — Import libraries
# Streamlit creates the web interface.
# PyTorch runs the neural network.
# Transformers loads the pretrained model and tokenizer.
# time.perf_counter() measures elapsed time precisely.

import json

from evaluation.streamlit_error_analysis import (
    analyze_translation,
)
from evaluation.streamlit_attempts import (
    save_attempt,
    get_attempt,
    save_scores,
)

from vocabulary_page import render_vocabulary_page
from pathlib import Path
import time
import streamlit as st
import torch
from human_preference_page import render_human_preference_page
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from human_analytics_page import render_human_analytics_page

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
from evaluation.comet_subprocess import calculate_comet_isolated

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

PROJECT_ROOT = Path(__file__).resolve().parent

TRANSLATION_MODELS = {
    "E0": {
        "name": "Baseline NMT",
        "path": "Helsinki-NLP/opus-mt-de-en",
        "type": "nmt",
    },
    "E1": {
        "name": "Technical NMT",
        "path": str(
            PROJECT_ROOT
            / "results/day8-de-en-v2/technical-finetuned/final_model"
        ),
        "type": "nmt",
    },
    "E2": {
        "name": "Mixed NMT",
        "path": str(
            PROJECT_ROOT
            / "results/day8-de-en-v2/mixed-finetuned/final_model"
        ),
        "type": "nmt",
    },
    "E3": {
        "name": "Qwen LLM",
        "path": "mlx-community/Qwen2.5-3B-Instruct-4bit",
        "type": "llm",
    },
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

@st.cache_resource
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



def translate_qwen(text: str):
    """
    Generate a German-to-English translation with Qwen
    using MLX on Apple Silicon.
    """
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    model_id = TRANSLATION_MODELS["E3"]["path"]

    model, tokenizer = load(model_id)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional German-to-English translator. "
                "Preserve terminology, numbers, placeholders, and negation. "
                "Return only the English translation."
            ),
        },
        {
            "role": "user",
            "content": text,
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    start = time.perf_counter()

    output_parts = []

    for response in stream_generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=128,
        sampler=make_sampler(temp=0),
    ):
        output_parts.append(response.text)

    elapsed_ms = (time.perf_counter() - start) * 1000

    return "".join(output_parts).strip(), elapsed_ms


def translate_with_model(text: str, model_key: str):
    """Translate using one of the four TransistorAI systems."""

    config = TRANSLATION_MODELS[model_key]

    if config["type"] == "llm":
        translation, latency = translate_qwen(text)

    else:
        tokenizer, model = load_translation_model(
            config["path"]
        )

        translation, latency = translate(
            text,
            tokenizer,
            model,
        )

    return {
        "model": model_key,
        "name": config["name"],
        "translation": translation,
        "latency_ms": latency,
    }


# STEP 6 — Configure the Streamlit interface
# The page title and layout define the user experience.

st.set_page_config(
    page_title="TransistorAI",
    page_icon="🌍",
    layout="centered",
)

st.title("🌍 TransistorAI")

# STEP 6A — Create the five main application tabs.
# Each tab provides an independent TransistorAI feature.

(
    translate_tab,
    practice_tab,
    vocabulary_tab,
    preference_tab,
    human_analytics_tab,
) = st.tabs([
    "🌍 Translate",
    "🎓 Test Yourself",
    "📚 Daily Vocabulary",
    "🏆 Human Preference Evaluation",
    "📊 Human Evaluation Analytics",
])




# ============================================================
# TAB 1 — TRANSLATE
# ============================================================


with translate_tab:

    st.subheader("German → English Translation")

    source_text = st.text_area(
        "Enter German text:",
        placeholder="Enter your German text here...",
        height=120,
        key="translation_source",
    )

    comparison_mode = st.radio(
        "Translation mode",
        ["Single model", "Compare all models"],
        horizontal=True,
    )

    model_labels = {
        key: f"{key} — {config['name']}"
        for key, config in TRANSLATION_MODELS.items()
    }

    selected_model = None

    if comparison_mode == "Single model":
        selected_model = st.selectbox(
            "Choose translation model",
            options=list(TRANSLATION_MODELS.keys()),
            format_func=lambda key: model_labels[key],
        )

    if st.button("Translate", type="primary", key="translate_models"):

        if not source_text.strip():
            st.warning("Please enter some German text.")

        else:
            models_to_run = (
                [selected_model]
                if comparison_mode == "Single model"
                else list(TRANSLATION_MODELS.keys())
            )

            results = []
            errors = []

            for model_key in models_to_run:
                with st.spinner(
                    f"Generating translation with {model_labels[model_key]}..."
                ):
                    try:
                        result = translate_with_model(
                            source_text,
                            model_key,
                        )

                        results.append(result)

                        log_translation(
                            source=source_text,
                            translation=result["translation"],
                            model_id=TRANSLATION_MODELS[model_key]["path"],
                            language_pair=LANGUAGE_PAIR,
                            latency_ms=result["latency_ms"],
                        )

                    except Exception as exc:
                        errors.append(
                            f"{model_labels[model_key]}: {exc}"
                        )

            st.session_state["translation_results"] = results
            st.session_state["translation_errors"] = errors
            st.session_state["translation_input"] = source_text

    results = st.session_state.get("translation_results", [])
    errors = st.session_state.get("translation_errors", [])

    if results:

        st.success(
            f"Generated {len(results)} translation(s) successfully."
        )

        st.markdown("### German source")
        st.write(st.session_state["translation_input"])

        st.markdown("### English translations")

        for result in results:

            with st.container(border=True):
                st.markdown(
                    f"#### {result['model']} — {result['name']}"
                )

                st.write(result["translation"])

                st.caption(
                    f"Latency: {result['latency_ms']:.2f} ms"
                )

        if len(results) > 1:

            st.markdown("### Model comparison")

            comparison_df = pd.DataFrame(
                [
                    {
                        "Model": f"{r['model']} — {r['name']}",
                        "Translation": r["translation"],
                        "Latency (ms)": round(r["latency_ms"], 2),
                    }
                    for r in results
                ]
            )

            st.dataframe(
                comparison_df,
                hide_index=True,
                use_container_width=True,
            )

        with st.expander("Decoding configuration"):
            st.json(DECODING_CONFIG)
            st.caption(
                "These decoding settings apply to the NMT models. "
                "Qwen uses separate MLX generation settings."
            )

    for error in errors:
        st.error(error)



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
    
    
    
    # ========================================================
    # STAGE 1 — SUBMIT AND ANALYZE
    # ========================================================

    exercise_key = f"{domain}_{exercise['id']}"
    state_key = "practice_active_attempt"

    if st.button("Analyze translation", type="primary"):

        if not user_answer.strip():
            st.warning("Please enter a translation.")

        else:
            source = str(exercise["source"])
            reference = str(exercise["reference"])

            analysis = analyze_translation(
                source=source,
                reference=reference,
                prediction=user_answer,
            )

            attempt_id = save_attempt(
                exercise_id=exercise["id"],
                domain=domain,
                source=source,
                reference=reference,
                user_translation=user_answer,
                analysis=analysis,
            )

            st.session_state[state_key] = {
                "attempt_id": attempt_id,
                "exercise_key": exercise_key,
                "enable_comet": enable_comet,
            }

    # ========================================================
    # STAGE 2 — SHOW ERROR ANALYSIS
    # ========================================================

    active = st.session_state.get(state_key)

    if active and active["exercise_key"] == exercise_key:

        attempt = get_attempt(active["attempt_id"])

        if attempt is None:
            st.error("Saved attempt could not be found.")
            st.stop()

        st.markdown("### Translation error analysis")

        st.write("**German source:**", attempt["source"])
        st.write("**Your translation:**", attempt["user_translation"])
        st.write("**Reference translation:**", attempt["reference"])

        analysis = json.loads(attempt["rule_feedback"])

        status_labels = {
            "review_suggested": "Review suggested",
            "no_difference_detected": "No difference detected",
            "not_verified": "Not automatically verified",
        }

        for item in analysis:

            with st.container(border=True):

                st.markdown(
                    f"**{item['category'].replace('_', ' ').title()}**"
                )

                st.caption(
                    status_labels.get(
                        item["status"],
                        item["status"],
                    )
                )

                st.write(item["explanation"])

                st.caption(f"Rule: {item['rule']}")

        st.info(
            "These are heuristic checks, not verified errors. "
            "Different wording may be equally correct."
        )

        # ====================================================
        # STAGE 3 — REVEAL METRICS
        # ====================================================

        if attempt["scores_revealed_at"] is None:

            if st.button(
                "Reveal evaluation scores",
                type="primary",
                key=f"reveal_{attempt['attempt_id']}",
            ):

                with st.spinner("Generating model translation..."):

                    tokenizer, model = load_translation_model(
                        MODELS["de-en"]
                    )

                    model_answer, latency_ms = translate(
                        attempt["source"],
                        tokenizer,
                        model,
                    )

                user_scores = calculate_translation_metrics(
                    attempt["user_translation"],
                    attempt["reference"],
                )

                model_scores = calculate_translation_metrics(
                    model_answer,
                    attempt["reference"],
                )

                comet_scores = None

                if active["enable_comet"]:

                    try:
                        with st.spinner("Calculating COMET scores..."):

                            comet_scores = calculate_comet_isolated(
                                source=attempt["source"],
                                user_translation=attempt["user_translation"],
                                model_translation=model_answer,
                                reference=attempt["reference"],
                            )

                    except Exception as error:
                        st.warning(
                            f"COMET unavailable: {error}"
                        )

                save_scores(
                    attempt_id=attempt["attempt_id"],
                    user_scores=user_scores,
                    model_scores=model_scores,
                    model_translation=model_answer,
                    comet_scores=comet_scores,
                )

                st.rerun()

        else:

            # =================================================
            # STAGE 4 — DISPLAY SAVED SCORES
            # =================================================

            st.markdown("### Translation comparison")

            st.write(
                "**Model translation:**",
                attempt["model_translation"],
            )

            results = pd.DataFrame({
                "Metric": ["BLEU", "chrF"],
                "Your translation": [
                    attempt["bleu"],
                    attempt["chrf"],
                ],
                "Model translation": [
                    attempt["model_bleu"],
                    attempt["model_chrf"],
                ],
            })

            if attempt["comet"] is not None:

                results = pd.concat([
                    results,
                    pd.DataFrame([{
                        "Metric": "COMET",
                        "Your translation": round(
                            attempt["comet"], 4
                        ),
                        "Model translation": round(
                            attempt["model_comet"], 4
                        ),
                    }]),
                ], ignore_index=True)

            st.markdown("### Evaluation metrics")
            st.dataframe(
                results,
                hide_index=True,
                use_container_width=True,
            )

            st.markdown("#### BLEU and chrF")

            st.bar_chart(
                results[
                    results["Metric"] == "COMET"
                ].set_index("Metric"),
                stack=False,
                y_label="COMET score",
            )

            if attempt["comet"] is not None:

                st.markdown("#### COMET")

                st.bar_chart(
                    results[
                        results["Metric"] == "COMET"
                    ].set_index("Metric")
                )

            st.caption(
                "BLEU and chrF use a 0–100 scale. "
                "COMET is a neural quality estimate "
                "and uses a different scale."
            )



# ============================================================
# TAB 3 — DAILY GERMAN VOCABULARY
# ============================================================

with vocabulary_tab:
    render_vocabulary_page()


# ============================================================
# TAB 4 — HUMAN PREFERENCE EVALUATION
# ============================================================

with preference_tab:
    render_human_preference_page()

# ============================================================
# TAB 5 — HUMAN EVALUATION ANALYTICS
# ============================================================
with human_analytics_tab:
    render_human_analytics_page()