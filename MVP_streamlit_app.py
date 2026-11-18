# STEP 1 — Import libraries
# Streamlit creates the web interface.
# PyTorch runs the neural network.
# Transformers loads the pretrained model and tokenizer.
# time.perf_counter() measures elapsed time precisely.
import altair as alt
from html import escape
import json
from project_explorer_page import render_project_explorer_page

from evaluation.hybrid_error_analysis import analyze_hybrid

from evaluation.streamlit_error_analysis import (
    analyze_translation,
    detect_mistakes,
)

from evaluation.mistake_tracker import save_mistakes

from evaluation.streamlit_attempts import (
    save_attempt,
    get_attempt,
    save_scores,
)

from user_selection_page import (
    render_user_selection,
    render_active_user_sidebar,
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

from my_mistakes_page import render_my_mistakes_page

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
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# APPLICATION NAVIGATION
# ============================================================

if "app_section" not in st.session_state:
    st.session_state.app_section = "welcome"

section = st.session_state.app_section


if section == "welcome":

    st.markdown("""
    <style>
    /* Decorative bilingual background */
    .welcome-background {
        position: fixed;
        inset: 0;
        overflow: hidden;
        pointer-events: none;
        z-index: 0;
    }

    .welcome-word {
        position: absolute;
        color: rgba(150, 170, 200, 0.12);
        font-weight: 700;
        white-space: nowrap;
        user-select: none;
    }

    .word-1 { top: 20%; left: 7%; font-size: 5rem; transform: rotate(-14deg); }
    .word-2 { top: 38%; right: 7%; font-size: 4rem; transform: rotate(12deg); }
    .word-3 { top: 65%; left: 5%; font-size: 4rem; transform: rotate(-10deg); }
    .word-4 { top: 78%; right: 6%; font-size: 4rem; transform: rotate(10deg); }
    .word-5 { top: 12%; right: 12%; font-size: 2.5rem; transform: rotate(8deg); }
    .word-6 { top: 86%; left: 25%; font-size: 3rem; transform: rotate(-6deg); }

    /* Welcome title */
    .welcome-heading {
        text-align: center;
        position: relative;
        z-index: 1;
        padding: 2rem 0 1.5rem;
    }

    .welcome-heading h1 {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .welcome-heading p {
        color: #9CA3AF;
        font-size: 1.05rem;
    }


    /* Blue Interaction button */
    .st-key-welcome_interaction button {
        background-color: #3B82F6 !important;
        border-color: #3B82F6 !important;
        color: white !important;
    }

    .st-key-welcome_interaction button:hover {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
    }

    /* Card text */
    .welcome-card-heading {
        text-align: center;
        font-size: 1.6rem;
        font-weight: 750;
        margin-bottom: 0.4rem;
    }

    .welcome-card-subtitle {
        text-align: center;
        font-size: 1.1rem;
        font-weight: 650;
        margin-bottom: 0.8rem;
    }

    .welcome-card-description {
        text-align: center;
        color: #AEB8C7;
        line-height: 1.6;
        margin-bottom: 1rem;
    }

    @media (max-width: 768px) {
        .welcome-word {
            font-size: 2rem !important;
        }
        .welcome-heading h1 {
            font-size: 2rem;
        }
    }
    </style>

    <div class="welcome-background" aria-hidden="true">
        <span class="welcome-word word-1">Hello</span>
        <span class="welcome-word word-2">Hallo</span>
        <span class="welcome-word word-3">Welcome</span>
        <span class="welcome-word word-4">Willkommen</span>
        <span class="welcome-word word-5">Good morning</span>
        <span class="welcome-word word-6">Guten Morgen</span>
    </div>

    <div class="welcome-heading">
        <h1>
            🌍 Welcome to Transistor<span style="color: #FF4B4B;">A</span><span style="color: #3B82F6;">I</span>
        </h1>
        <p>Explore translation research. Experience machine learning.</p>
    </div>
    """, unsafe_allow_html=True)

    # Center the navigation cards.
    left, center, right = st.columns([1, 2.4, 1])

    with center:

        # Card 1 — Analytics
        with st.container(border=True):
            st.markdown("""
            <div class="welcome-card-heading">
                📊 Transistor<span style="color: #FF4B4B;">Analytics</span>
            </div>
            <div class="welcome-card-subtitle">
                Machine Learning Translation Research Project
            </div>
            <div class="welcome-card-description">
                Discover the project architecture, datasets,
                training process, model comparisons and
                human evaluation results.
            </div>
            """, unsafe_allow_html=True)

            if st.button(
                "Open TransistorAnalytics",
                use_container_width=True,
                type="primary",
                key="welcome_analytics",
            ):
                st.session_state.app_section = "analytics"
                st.rerun()

        st.markdown("<div style='height: 90px;'></div>", unsafe_allow_html=True)

        # Card 2 — Interaction
        with st.container(border=True):
            st.markdown("""
            <div class="welcome-card-heading">
                🌍 Transistor<span style="color: #3B82F6;">Interaction</span>
            </div>
            <div class="welcome-card-subtitle">
                Let's translate together and learn!
            </div>
            <div class="welcome-card-description">
                Translate using different models, learn vocabulary
                and participate in human preference evaluation.
            </div>
            """, unsafe_allow_html=True)

            if st.button(
                "Open TransistorInteraction",
                use_container_width=True,
                type="primary",
                key="welcome_interaction",
            ):
                st.session_state.app_section = "interaction"
                st.rerun()

    st.stop()
    st.title("🌍 Welcome to TransistorAI")

    st.markdown(
        "Explore the research behind TransistorAI or interact "
        "with our German-to-English translation systems."
    )

    analytics_col, interaction_col = st.columns(2, gap="large")

    with analytics_col:
        with st.container(border=True):
            st.subheader("📊 TransistorAnalytics")
            st.write(
                "Discover the project architecture, datasets, "
                "training process, model comparisons and "
                "human evaluation results."
            )

            if st.button(
                "Open TransistorAnalytics",
                use_container_width=True,
                type="primary",
            ):
                st.session_state.app_section = "analytics"
                st.rerun()

    with interaction_col:
        with st.container(border=True):
            st.subheader("🌍 TransistorInteraction")
            st.write(
                "Translate text, practice German-to-English "
                "translation, learn vocabulary and participate "
                "in human preference evaluation."
            )

            if st.button(
                "Open TransistorInteraction",
                use_container_width=True,
                type="primary",
            ):
                st.session_state.app_section = "interaction"
                st.rerun()

    st.stop()


# Shared navigation
with st.sidebar:
    st.markdown("### 🌍 TransistorAI")

    if st.button("🏠 Home", use_container_width=True):
        st.session_state.app_section = "welcome"
        st.rerun()

    if st.button("📊 Analytics", use_container_width=True):
        st.session_state.app_section = "analytics"
        st.rerun()

    if st.button("🌍 Interaction", use_container_width=True):
        st.session_state.app_section = "interaction"
        st.rerun()


# ============================================================
# TRANSISTOR ANALYTICS
# ============================================================

if section == "analytics":
    st.title("📊 TransistorAnalytics")

    project_explorer_tab, human_analytics_tab = st.tabs([
        "📖 Project Explorer",
        "📊 Human Evaluation Analytics",
    ])

    with project_explorer_tab:
        render_project_explorer_page()

    with human_analytics_tab:
        render_human_analytics_page()

    st.stop()


# ============================================================
# TRANSISTOR INTERACTION
# ============================================================

if section == "interaction":
    active_user = render_user_selection()

    if active_user is None:
        st.stop()

    active_user_id = active_user["user_id"]
    active_username = active_user["username"]

    render_active_user_sidebar()

    st.title("🌍 TransistorInteraction")

    (
        translate_tab,
        practice_tab,
        mistakes_tab,
        vocabulary_tab,
        preference_tab,
    ) = st.tabs([
        "🌍 Translate",
        "🎓 Test Yourself",
        "🧠 My Mistakes",
        "📚 Daily Vocabulary",
        "🏆 Human Preference Evaluation",
    ])




# ============================================================
# TAB 2 — TRANSLATE
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

            translation_columns = st.columns(
                4,
                gap="medium",
            )
            MODEL_COLORS = {
                "E0": "#E9A6A0",
                "E1": "#D99A9A",
                "E2": "#EB807B",
                "E3": "#D88D91",
            }
            for index, result in enumerate(results):
                model_key = result["model"]
                color = MODEL_COLORS.get(model_key, "#E9A6A0")


                with translation_columns[index % 4]:

                    with st.container(border=True):

                        st.markdown(
                            f'<h4 style="color: {color};">'
                            f'{escape(model_key)} — {escape(result["name"])}'
                            f'</h4>',
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f'<p style="color: {color};">'
                            f'{escape(str(result["translation"]))}'
                            f'</p>',
                            unsafe_allow_html=True,
                        )

                        st.markdown(
                            f'<p style="color: {color}; font-size: 0.8rem;">'
                            f'Latency: {result["latency_ms"]:.2f} ms'
                            f'</p>',
                            unsafe_allow_html=True,
                        )




            if len(results) > 1:

                st.markdown("### Model comparison")

                comparison_df = pd.DataFrame([
                    {
                        "Model": f"{r['model']} — {r['name']}",
                        "Translation": r["translation"],
                        "Latency (ms)": round(r["latency_ms"], 2),
                    }
                    for r in results
                ])


                # Styled HTML table with larger font sizes
                table_rows = []

                for result in results:
                    model_key = result["model"]
                    color = MODEL_COLORS.get(model_key, "#FFFFFF")

                    model_name = escape(
                        f'{model_key} — {result["name"]}'
                    )
                    translation = escape(
                        str(result["translation"])
                    )
                    latency = f'{result["latency_ms"]:.2f}'

                    table_rows.append(
                        f'<tr style="color: {color};">'
                        f'<td>{model_name}</td>'
                        f'<td>{translation}</td>'
                        f'<td style="text-align: right;">{latency}</td>'
                        f'</tr>'
                    )

                table_html = f"""
                <style>
                .model-comparison-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 16px;
                    line-height: 1.5;
                }}

                .model-comparison-table th {{
                    text-align: left;
                    padding: 12px;
                    font-size: 16px;
                    color: #D1D5DB;
                    background: #1B1E26;
                    border: 1px solid #343943;
                }}

                .model-comparison-table td {{
                    padding: 12px;
                    border: 1px solid #343943;
                    vertical-align: middle;
                }}

                .model-comparison-table tr:hover {{
                    background: #1A2029;
                }}
                </style>

                <table class="model-comparison-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Translation</th>
                            <th style="text-align: right;">Latency (ms)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {''.join(table_rows)}
                    </tbody>
                </table>
                """

                st.markdown(
                    table_html,
                    unsafe_allow_html=True,
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
# TAB 3 — TEST YOURSELF SECTION
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

        exercise_col, translation_col = st.columns(
            2,
            gap="large",
        )

        # LEFT — Exercise selection and German source
        with exercise_col:

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

        # RIGHT — English translation and COMET
        with translation_col:

            user_answer = st.text_area(
                "Your English translation:",
                key=(
                    f"answer_{active_user_id}_"
                    f"{domain}_{exercise['id']}"
                ),
                height=150,
                placeholder="Write your English translation here...",
            )

            enable_comet = st.checkbox(
                "Calculate COMET",
                value=True,
                help=(
                    "Evaluate translation quality "
                    "using a neural model."
                ),
            )



        # ========================================================
        # STAGE 1 — SUBMIT AND ANALYZE
        # ========================================================

        exercise_key = f"{domain}_{exercise['id']}"
        state_key = f"practice_active_attempt_{active_user_id}"

        with translation_col:
            analyze_clicked = st.button(
                "Analyze translation",
                type="primary",
            )

        if analyze_clicked:

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
                    user_id=active_user_id,
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

            attempt = get_attempt(
                active["attempt_id"],
                user_id=active_user_id,
            )

            if attempt is None:
                st.error("Saved attempt could not be found.")
                st.stop()

            st.markdown("### Translation error analysis")




            st.markdown(
                f'<p style="color: #E9A6A0; margin-bottom: 8px;">'
                f'<strong>German source:</strong> {escape(str(attempt["source"]))}'
                f'</p>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f'<p style="color: #E9A6A0; margin-bottom: 8px;">'
                f'<strong>Reference translation:</strong> {escape(str(attempt["reference"]))}'
                f'</p>',
                unsafe_allow_html=True,
            )

            st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

            st.markdown(
                f'<p style="color: #EB807B;">'
                f'<strong>Your translation:</strong> {escape(str(attempt["user_translation"]))}'
                f'</p>',
                unsafe_allow_html=True,
            )


            analysis = json.loads(attempt["rule_feedback"])


            # ====================================================
            # INDIVIDUAL MISTAKE DETECTION AND SELECTION
            # ====================================================


            analysis_key = f"hybrid_analysis_{attempt['attempt_id']}"

            if analysis_key not in st.session_state:
                with st.spinner(
                    "Analyzing your translation with Qwen and Python rules..."
                ):
                    st.session_state[analysis_key] = analyze_hybrid(
                        source=attempt["source"],
                        reference=attempt["reference"],
                        prediction=attempt["user_translation"],
                    )

            hybrid_result = st.session_state[analysis_key]
            mistakes = hybrid_result["mistakes"]


            general_feedback = hybrid_result.get("general_feedback", [])

            if general_feedback:
                st.markdown("#### Translation feedback")

                st.caption(
                    "These are AI-generated observations that need review. "
                    "They are not confirmed mistakes."
                )

                for feedback in general_feedback:
                    with st.container(border=True):
                        st.write(feedback["explanation"])

                        st.markdown("**Your translation**")
                        st.write(feedback["original_text"])

                        st.markdown("**Suggested alternative**")
                        st.write(feedback["corrected_text"])


            if hybrid_result["warning"]:
                st.warning(hybrid_result["warning"])

            if not mistakes:
                st.success(
                    "No differences detected by the current rules. "
                    "This does not guarantee a perfect translation."
                )

            else:
                st.markdown(
                    f"#### {len(mistakes)} potential mistake(s)"
                )

                st.caption(
                    "Review each suggestion before saving. "
                    "Alternative translations may also be correct."
                )

                selected_mistakes = []

                for index, mistake in enumerate(mistakes):

                    with st.container(border=True):

                        selected = st.checkbox(
                            f"Track: {mistake['category'].replace('_', ' ').title()}",
                            key=(
                                f"mistake_{attempt['attempt_id']}_{index}"
                            ),
                            value=False,
                        )

                        left, right = st.columns(2)

                        with left:
                            st.markdown("**Your text**")
                            st.write(
                                mistake["original_text"] or "—"
                            )

                        with right:
                            st.markdown("**Suggested correction**")
                            st.write(
                                mistake["corrected_text"] or "—"
                            )

                        st.caption(mistake["explanation"])

                        if selected:
                            selected_mistakes.append(mistake)

                if st.button(
                    "Save selected mistakes",
                    type="primary",
                    key=f"save_mistakes_{attempt['attempt_id']}",
                    disabled=not selected_mistakes,
                ):
                    saved_ids = save_mistakes(
                        attempt_id=attempt["attempt_id"],
                        user_id=active_user_id,
                        mistakes=selected_mistakes,
                    )

                    st.success(
                        f"Saved {len(saved_ids)} mistake(s) "
                        "to your learning history."
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
                        user_id=active_user_id,
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


                # ================================================
                # BLEU AND chrF — FIXED SCALE, NO SCROLL ZOOM
                # ================================================

                st.markdown("#### BLEU and chrF")

                bleu_chrf_df = results[
                    results["Metric"].isin(["BLEU", "chrF"])
                ].melt(
                    id_vars="Metric",
                    value_vars=[
                        "Your translation",
                        "Model translation",
                    ],
                    var_name="Translation",
                    value_name="Score",
                )

                bleu_chrf_chart = (
                    alt.Chart(bleu_chrf_df)
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "Metric:N",
                            title=None,
                            sort=["BLEU", "chrF"],
                            axis=alt.Axis(labelAngle=0),
                        ),
                        xOffset=alt.XOffset("Translation:N"),
                        y=alt.Y(
                            "Score:Q",
                            title="Score (0–100)",
                            scale=alt.Scale(
                                domain=[0, 100],
                                nice=False,
                            ),
                        ),
                        color=alt.Color(
                            "Translation:N",
                            title="Translation",
                        ),
                        tooltip=[
                            alt.Tooltip("Metric:N"),
                            alt.Tooltip("Translation:N"),
                            alt.Tooltip("Score:Q", format=".2f"),
                        ],
                    )
                    .properties(height=300)

                )

                st.altair_chart(
                    bleu_chrf_chart,
                    use_container_width=True,
                )

                # ================================================
                # COMET — FIXED SCALE, NO SCROLL ZOOM
                # ================================================

                if attempt["comet"] is not None:

                    st.markdown("#### COMET")

                    comet_df = results[
                        results["Metric"] == "COMET"
                    ].melt(
                        id_vars="Metric",
                        value_vars=[
                            "Your translation",
                            "Model translation",
                        ],
                        var_name="Translation",
                        value_name="Score",
                    )

                    comet_chart = (
                        alt.Chart(comet_df)
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "Metric:N",
                                title=None,
                                axis=alt.Axis(labelAngle=0),
                            ),
                            xOffset=alt.XOffset("Translation:N"),
                            y=alt.Y(
                                "Score:Q",
                                title="COMET score",
                                scale=alt.Scale(
                                    domain=[0, 1],
                                    nice=False,
                                    zero=True,
                                ),
                                axis=alt.Axis(
                                    values=[0, 0.2, 0.4, 0.6, 0.8, 1],
                                ),
                            ),
                            color=alt.Color(
                                "Translation:N",
                                title="Translation",
                            ),
                            tooltip=[
                                alt.Tooltip("Translation:N"),
                                alt.Tooltip("Score:Q", format=".4f"),
                            ],
                        )
                        .properties(height=300)

                    )

                    st.altair_chart(
                        comet_chart,
                        use_container_width=True,
                    )




    # ========================================================
    # TAB 3 — DAILY VOCABULARY
    # ========================================================

    with vocabulary_tab:
        render_vocabulary_page(learner_id=active_user_id)

    # ========================================================
    # TAB 4 — HUMAN PREFERENCE EVALUATION
    # ========================================================

    with preference_tab:
        render_human_preference_page(
            user_id=active_user_id,
            username=active_username,
        )

    # ============================================================
    # MY MISTAKES
    # ============================================================

    with mistakes_tab:
        render_my_mistakes_page(
            user_id=active_user_id
        )