import streamlit as st
import pandas as pd
from pathlib import Path
import sacrebleu

PROJECT_ROOT = Path(__file__).resolve().parent


def render_project_explorer_page():

    st.title("📖 TransistorAI — Project Explorer")

    st.markdown(
        """
        Explore how TransistorAI was developed, from dataset
        preparation and model training to translation quality
        evaluation and human preferences.
        """
    )

    metrics_tab, data_tab, models_tab, experiments_tab = st.tabs([
        "📊 Evaluation Metrics",
        "🗂️ Data & Cleaning",
        "🧠 Models",
        "🔬 Experiments & Results",
    ])

# ============================================================
# Tab 1 — Evaluation Metrics
# ============================================================
    with metrics_tab:
        st.header("📊 Translation Evaluation Metrics")

        st.markdown("""
        Translation quality is difficult to measure automatically
        because a sentence can have several equally correct translations.

        TransistorAI uses three complementary metrics:

        - **BLEU** — measures matching word sequences.
        - **chrF** — measures matching character sequences.
        - **COMET** — uses a trained neural model to assess translation quality.

        Higher scores generally indicate better translation quality,
        but no metric perfectly captures human judgment.
        """)

        bleu_tab, chrf_tab, comet_tab = st.tabs([
                "BLEU", "chrF", "COMET"
        ])

        with bleu_tab:
            st.subheader("BLEU — Bilingual Evaluation Understudy")

            st.markdown("""
            BLEU measures how many word sequences (n-grams)
            in a translation also appear in a reference translation.

            It typically uses 1-grams through 4-grams.

            **Example:**

            Reference: `The system is working correctly.`

            Prediction: `The system works correctly.`

            Both sentences express similar meanings, but BLEU
            penalizes differences in word sequences.

            BLEU also includes a **brevity penalty** to discourage
            translations that are too short.

            **Strength:** Useful for standardized corpus-level comparisons.

            **Limitation:** Can penalize valid paraphrases and synonyms.
            """)
            st.divider()
            st.subheader("🧪 Interactive Translation Metrics Calculator")

            st.markdown("""
            Experiment with different translations to understand
            how BLEU and chrF respond to wording changes.

            Try replacing a word with a synonym, removing a word,
            or changing the sentence structure.
            """)

            reference = st.text_area(
                "Reference translation",
                value="The system is working correctly.",
                key="explorer_reference",
            )

            prediction = st.text_area(
                "Candidate translation",
                value="The system works correctly.",
                key="explorer_prediction",
            )

            if st.button("Calculate Metrics", key="explorer_calculate"):

                if not reference.strip() or not prediction.strip():
                    st.warning("Please enter both translations.")

                else:
                    bleu = sacrebleu.sentence_bleu(
                        prediction,
                        [reference],
                    ).score

                    chrf = sacrebleu.sentence_chrf(
                        prediction,
                        [reference],
                    ).score

                    col1, col2 = st.columns(2)

                    with col1:
                        st.metric("BLEU", f"{bleu:.2f}")

                    with col2:
                        st.metric("chrF", f"{chrf:.2f}")

                    chart_data = pd.DataFrame({
                        "Metric": ["BLEU", "chrF"],
                        "Score": [bleu, chrf],
                    })

                    st.bar_chart(
                        chart_data.set_index("Metric"),
                        y_label="Score (0–100)",
                    )

                    st.info(
                        "These are sentence-level scores. "
                        "Corpus-level BLEU and chrF are calculated "
                        "across multiple sentences and are not "
                        "simply averages of sentence scores."
                    )

                    st.caption(
                        "COMET is not calculated here because it "
                        "requires a neural evaluation model and "
                        "the original source sentence."
                    )
        with chrf_tab:
            st.subheader("chrF — Character F-score")

            st.markdown("""
            chrF compares character sequences rather than complete words.

            It combines character n-gram precision and recall
            into an F-score.

            **Example:**

            Reference: `The connection was established.`

            Prediction: `The connection is established.`

            Many character sequences overlap, even though
            the sentences are not identical.

            **Strength:** Can recognize similarities in word forms
            that word-based metrics miss.

            **Limitation:** Character similarity does not guarantee
            semantic correctness.
            """)

        with comet_tab:
            st.subheader("COMET — Neural Translation Evaluation")

            st.markdown("""
            COMET uses a pretrained neural evaluation model to
            estimate translation quality.

            In TransistorAI, we use:

            `Unbabel/wmt22-comet-da`

            The model considers:

            1. The original German sentence.
            2. The generated English translation.
            3. The English reference translation.

            COMET can recognize semantic similarities that
            BLEU and chrF may penalize.

            **Strength:** Often aligns better with human quality judgments.

            **Limitation:** More computationally expensive and can
            still make incorrect judgments.

            COMET scores are not percentages and should not be
            directly compared numerically with BLEU or chrF.
            """)
# ============================================================
# TAB 2 — DATASETS AND DATA PREPARATION
# ============================================================
    with data_tab:
        st.header("🗂️ Datasets and Data Preparation")

        st.markdown("""
        TransistorAI uses two German-to-English translation
        domains to investigate how fine-tuning affects
        translation quality.

        **Technical domain — KDE4**

        Software interface messages and technical terminology,
        including application settings, commands and placeholders.

        **General domain — Tatoeba/OPUS**

        General-language sentence pairs used to evaluate
        translation performance outside the technical domain.
        """)

        dataset_df = pd.DataFrame({
            "Split": ["Train", "Validation", "Test"],
            "Technical": [101401, 12662, 12623],
            "General": [25001, 2499, 1997],
        })

        st.subheader("Dataset distribution")

        st.dataframe(
            dataset_df,
            hide_index=True,
            use_container_width=True,
        )

        st.bar_chart(
            dataset_df.set_index("Split"),
            y_label="Sentence pairs",
        )

        st.metric(
            "Total sentence pairs across splits",
            f"{dataset_df[['Technical', 'General']].sum().sum():,}",
        )
        st.divider()
        st.subheader("Data Preparation Pipeline")

        st.markdown("""
        Before training, the parallel translation data
        must be prepared so that each German source
        sentence corresponds to its correct English reference.
        """)

        pipeline = [
            ("1. Data collection",
             "Load German–English sentence pairs from KDE4 and Tatoeba/OPUS."),

            ("2. Data inspection",
             "Inspect source and reference text, dataset sizes and data quality."),

            ("3. Cleaning",
             "Apply the project's preprocessing and quality-control rules."),

            ("4. Dataset splitting",
             "Organize examples into training, validation and test datasets."),

            ("5. Tokenization",
             "Convert sentences into token IDs using the model tokenizer."),

            ("6. Model training",
             "Use the prepared training data to fine-tune the translation models."),

            ("7. Evaluation",
             "Evaluate translations against held-out references using BLEU, chrF and COMET."),
        ]

        for title, description in pipeline:
            with st.expander(title):
                st.write(description)


    
    with models_tab:
        st.header("🧠 Translation Models")

        st.markdown("""
        TransistorAI compares four German-to-English translation
        systems: three Neural Machine Translation (NMT) systems
        based on MarianMT, and one Large Language Model (LLM)
        based on Qwen.

        The experiments investigate whether domain-specific
        fine-tuning improves translation quality and how a
        general-purpose LLM compares with specialized NMT models.
        """)

        st.subheader("Model comparison")

        model_df = pd.DataFrame([
            {
                "Model": "E0",
                "Name": "Baseline NMT",
                "Architecture": "MarianMT",
                "Training": "Pretrained",
                "Data": "Original pretrained model",
            },
            {
                "Model": "E1",
                "Name": "Technical NMT",
                "Architecture": "MarianMT",
                "Training": "Fine-tuned",
                "Data": "KDE4 technical",
            },
            {
                "Model": "E2",
                "Name": "Mixed NMT",
                "Architecture": "MarianMT",
                "Training": "Fine-tuned",
                "Data": "Technical + General",
            },
            {
                "Model": "E3",
                "Name": "Qwen LLM",
                "Architecture": "Decoder-only Transformer",
                "Training": "Prompted inference",
                "Data": "No project-specific fine-tuning",
            },
        ])

        st.dataframe(
            model_df,
            hide_index=True,
            use_container_width=True,
        )

        st.divider()
        st.subheader("Understanding the Architectures")

        nmt_arch, llm_arch = st.tabs([
            "MarianMT — Encoder–Decoder",
            "Qwen — Decoder-only LLM",
        ])

        with nmt_arch:
            st.markdown("""
            ### MarianMT

            MarianMT is a Transformer-based Neural Machine
            Translation model designed for translation tasks.

            It uses two main components:

            **Encoder**
            - Receives the German input sentence.
            - Converts tokens into contextual representations.
            - Captures relationships between words.

            **Decoder**
            - Generates the English translation.
            - Uses previously generated tokens.
            - Attends to the encoder representations.

            **Translation process:**

            German sentence → Tokenizer → Encoder →
            Decoder → English translation

            E0, E1 and E2 share this architecture.
            The difference is their fine-tuning history.
            """)

        with llm_arch:
            st.markdown("""
            ### Qwen2.5-3B-Instruct

            Qwen is a general-purpose Large Language Model
            based on a decoder-only Transformer architecture.

            Unlike MarianMT, it does not use a separate
            encoder to process the source sentence.

            Instead, it receives a prompt containing the
            translation instructions and German sentence,
            then generates the English output autoregressively.

            **Translation process:**

            Instructions + German sentence → Tokenizer →
            Decoder-only Transformer → English translation

            **TransistorAI configuration:**
            - Model: Qwen2.5-3B-Instruct
            - Approximately 3 billion parameters
            - 4-bit quantized MLX version
            - Temperature: 0
            - Maximum generation: 128 tokens

            The model is used as a prompted baseline,
            without additional fine-tuning in this project.
            """)

        st.divider()
        st.subheader("The Four Experiments")

        with st.expander("E0 — Baseline NMT", expanded=True):
            st.markdown("""
            **Model:** Helsinki-NLP/opus-mt-de-en

            E0 is the original pretrained MarianMT model.

            It provides a baseline for measuring whether
            fine-tuning improves translation quality.

            **Training in TransistorAI:** None.

            **Purpose:** Establish performance before
            domain adaptation.
            """)

        with st.expander("E1 — Technical Fine-tuning"):
            st.markdown("""
            **Base model:** E0 MarianMT

            **Fine-tuning data:** KDE4 technical translations.

            E1 adapts the pretrained model to software-related
            terminology, technical instructions and interface text.

            **Research question:**

            Does specializing a translation model improve
            technical translation quality at the expense
            of general-language performance?
            """)

        with st.expander("E2 — Mixed Fine-tuning"):
            st.markdown("""
            **Base model:** E0 MarianMT

            **Fine-tuning data:** Technical and general translations.

            E2 investigates whether training with both domains
            can improve technical performance while preserving
            more general-language translation ability.

            **Research question:**

            Can mixed-domain fine-tuning reduce the trade-off
            between technical specialization and generalization?
            """)

        with st.expander("E3 — Qwen LLM"):
            st.markdown("""
            **Model:** mlx-community/Qwen2.5-3B-Instruct-4bit

            E3 uses an instruction-following Large Language Model.

            The system prompt instructs the model to:
            - Translate German into English.
            - Preserve technical terminology.
            - Preserve numbers and placeholders.
            - Preserve negation.
            - Return only the translation.

            **Training in TransistorAI:** None.

            **Research question:**

            Can a general-purpose LLM compete with specialized
            NMT models on German-to-English translation?
            """)

        st.divider()
        st.subheader("Fine-tuning Configuration")

        config_df = pd.DataFrame([
            {"Parameter": "Learning rate", "Value": "2e-5"},
            {"Parameter": "Batch size per device", "Value": "4"},
            {"Parameter": "Gradient accumulation", "Value": "2"},
            {"Parameter": "Epochs", "Value": "3"},
            {"Parameter": "Maximum sequence length", "Value": "128"},
            {"Parameter": "Random seed", "Value": "42"},
        ])

        st.dataframe(
            config_df,
            hide_index=True,
            use_container_width=True,
        )

        st.caption(
            "Recorded MarianMT fine-tuning configuration. "
            "Confirm experiment-specific settings against the "
            "saved training configurations before final presentation."
        )


    
    with experiments_tab:
        st.header("🔬 Experiments & Results")

        st.markdown("""
        TransistorAI evaluates four translation systems across
        technical and general German-to-English translation.

        The goal is to understand how fine-tuning changes
        translation performance and how a general-purpose
        LLM compares with specialized NMT systems.
        """)

        results_path = (
            PROJECT_ROOT
            / "data/project_explorer/day14_pilot_metrics.csv"
        )

        if not results_path.exists():
            st.warning("Pilot results CSV not found.")
        else:
            results_df = pd.read_csv(results_path)

            st.subheader("Day 14 — LLM Comparison Pilot")

            st.info(
                "These results come from a pilot evaluation "
                "using 50 technical and 50 general sentences. "
                "They should not be confused with full-test "
                "results or interpreted as definitive rankings."
            )

            metric_labels = {
                "BLEU": "bleu",
                "chrF": "chrf",
                "COMET": "comet",
            }

            selected_metric = st.selectbox(
                "Choose evaluation metric",
                list(metric_labels.keys()),
                key="explorer_metric_selection",
            )

            metric_column = metric_labels[selected_metric]

            chart_df = results_df.pivot(
                index="model",
                columns="domain",
                values=metric_column,
            )

            st.bar_chart(
                chart_df,
                x_label="Translation system",
                y_label=f"{selected_metric} score",
                stack=False,
            )

            st.subheader("Detailed Results")

            st.dataframe(
                results_df,
                hide_index=True,
                use_container_width=True,
            )

            st.divider()
            st.subheader("What Did We Learn?")

            with st.expander(
                "Finding 1 — Technical fine-tuning",
                expanded=True,
            ):
                st.markdown("""
                E1 achieved the highest technical-domain
                BLEU, chrF and COMET scores in this pilot.

                This supports the hypothesis that fine-tuning
                on technical data improves performance in
                the target domain.
                """)

            with st.expander(
                "Finding 2 — General-domain performance"
            ):
                st.markdown("""
                E0 achieved the highest general-domain
                BLEU, chrF and COMET scores in this pilot.

                E1 performed substantially worse on general
                sentences than E0.

                This is consistent with a domain-specialization
                trade-off and possible catastrophic forgetting.
                However, the pilot alone does not establish
                the underlying cause.
                """)

            with st.expander(
                "Finding 3 — Mixed-domain fine-tuning"
            ):
                st.markdown("""
                E2 achieved technical performance close to E1,
                while preserving better general-domain performance
                than E1.

                This suggests mixed-domain training can provide
                a useful compromise between specialization
                and generalization.
                """)

            with st.expander(
                "Finding 4 — LLM comparison"
            ):
                st.markdown("""
                E3, the Qwen LLM, scored below the NMT systems
                on technical translations in this pilot.

                On general sentences, its COMET score was
                close to E0, despite lower BLEU and chrF.

                This highlights the importance of evaluating
                translation quality using complementary metrics.
                """)

            st.divider()
            st.subheader("Evaluation Limitations")

            st.markdown("""
            - Only 50 sentences per domain were included
              in the Day 14 pilot.
            - BLEU and chrF depend on reference wording.
            - COMET is a learned metric and is not infallible.
            - Automatic metrics do not replace human evaluation.
            - Model comparisons should use the same sentences
              and evaluation settings.
            - The LLM and NMT systems have different
              architectures and inference configurations.
            """)

            st.divider()
            st.subheader("Human Preferences vs Automatic Metrics")

            st.markdown("""
            During Day 15–17, TransistorAI introduced a
            blind human preference evaluation.

            Participants compare translations from E0,
            E1, E2 and E3 without seeing model identities.

            The project then compares their selections
            with BLEU, chrF and COMET rankings.
            """)

            st.markdown("#### Day 17 preliminary analysis")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Compared votes", "25")

            with col2:
                st.metric("Unique sentences", "21")

            with col3:
                st.metric("BLEU strict agreement", "36%")

            st.caption(
                "These values are from the preliminary Day 17 "
                "comparison output, not necessarily the full "
                "human preference database. Further coverage "
                "validation is required."
            )

            st.info(
                "A low agreement rate does not automatically "
                "mean a metric is unreliable. Sample size, "
                "ties, missing scores and evaluation design "
                "must also be considered."
            )
