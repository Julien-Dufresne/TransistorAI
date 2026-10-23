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

@st.cache_resource
def load_translation_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    model = model.to(DEVICE)
    model.eval()
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

    # STEP 7 — Load the model selected by LANGUAGE_PAIR.
    # The cached model is reused across Streamlit reruns.
    tokenizer, model = load_translation_model(MODEL_ID)

    # STEP 8 — Create the text input.
    # This variable contains the German sentence entered
    # by the user.
    source_text = st.text_area(
        "Enter German text:",
        value="Enter your text in German here",
        height=120,
    )

    # STEP 9 — Translate when the button is clicked.
    if st.button("Translate", type="primary"):

        # Prevent empty translations.
        if not source_text.strip():
            st.warning("Please enter some text.")

        else:
            with st.spinner("Translating..."):

                # STEP 10 — Generate the English translation.
                # The function returns the translated text
                # and the inference latency in milliseconds.
                translation, latency_ms = translate(
                    source_text,
                    tokenizer,
                    model,
                )

                # STEP 11 — Save the result to CSV.
                # This appends one row to translation_log.csv.
                log_translation(
                    source=source_text,
                    translation=translation,
                    model_id=MODEL_ID,
                    language_pair=LANGUAGE_PAIR,
                    latency_ms=latency_ms,
                )

            # STEP 12 — Display translation results.
            st.success("Translation completed!")

            st.markdown("### Source (German)")
            st.write(source_text)

            st.markdown("### Translation (English)")
            st.write(translation)

            st.markdown("### Model Information")
            st.write(f"**Model:** {MODEL_ID}")
            st.write(f"**Device:** {DEVICE}")
            st.write(f"**Latency:** {latency_ms:.2f} ms")

            # STEP 13 — Display decoding settings.
            with st.expander("Decoding configuration"):
                st.json(DECODING_CONFIG)

# ============================================================
# TAB 2 — TEST YOURSELF SECTION
# ============================================================

with practice_tab:

    st.subheader("Test Yourself — English → German")

    st.write(
        "Translate the English sentence into German, "
        "then compare your answer with the model."
    )

    # STEP 15 — Track which exercise the user is doing.
    # Session state preserves this value when Streamlit reruns.
    if "exercise_index" not in st.session_state:
        st.session_state.exercise_index = 0

    index = st.session_state.exercise_index
    exercise = EXERCISES[index]

    # Display the selected English sentence.
    st.write(f"Exercise {index + 1} of {len(EXERCISES)}")
    st.info(exercise["english"])

    # Show the grammar or vocabulary focus.
    st.caption(f"Focus: {exercise['focus']}")

    # STEP 16 — Capture the user's German translation.
    # The unique key creates a separate input for each exercise.
    user_answer = st.text_area(
        "Your German translation:",
        key=f"answer_{index}",
        height=100,
    )

    # STEP 17 — Check the user's translation.
    if st.button("Check my translation"):

        if not user_answer.strip():
            st.warning("Please enter a German translation.")

        else:
            with st.spinner("Evaluating your translation..."):

                # Load the English → German model.
                # This is independent of LANGUAGE_PAIR.
                test_tokenizer, test_model = load_translation_model(
                    MODELS["en-de"]
                )

                # Generate the model's German translation.
                model_answer, latency_ms = translate(
                    exercise["english"],
                    test_tokenizer,
                    test_model,
                )

                # Compare the user's answer with the model output.
                comparison = compare_translations(
                    user_answer,
                    model_answer,
                )

            # STEP 18 — Display comparison results.
            st.markdown("### Your translation")
            st.write(user_answer)

            st.markdown("### Model translation")
            st.write(model_answer)

            st.metric(
                "Word-sequence similarity",
                f"{comparison['similarity']}%",
            )

            # Explain what the comparison means.
            st.info(
                "This score measures similarity to the model's "
                "wording, not translation correctness. "
                "Different translations can both be valid."
            )

            st.markdown("### Differences")

            st.write("Words only in the model translation:")
            st.write(comparison["missing"])

            st.write("Words only in your translation:")
            st.write(comparison["extra"])

            # STEP 19 — Save the learning attempt.
            # Each attempt creates a new row in learning_history.csv.
            append_csv("learning_history.csv", {
                "timestamp_utc": datetime.now(
                    timezone.utc
                ).isoformat(),
                "exercise_id": exercise["id"],
                "english_source": exercise["english"],
                "user_translation": user_answer,
                "model_translation": model_answer,
                "focus": exercise["focus"],
                "similarity_percent": comparison["similarity"],
            })

    # STEP 20 — Move to the next exercise.
    if st.button("Next exercise"):

        # Increment the exercise index.
        # Modulo (%) returns to exercise 1 after exercise 5.
        st.session_state.exercise_index = (
            index + 1
        ) % len(EXERCISES)

        # Refresh the interface with the next exercise.
        st.rerun()

# STEP 21 — Application disclaimer.
# This appears below both tabs.

st.caption(
    "Research prototype. Translation quality is not "
    "guaranteed. Similarity is not a measure of correctness."
)