import streamlit as st
import pandas as pd
from pathlib import Path
import sacrebleu
import altair as alt
import json

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
    (
        structure_tab, training_tab, metrics_tab, data_tab, models_tab, experiments_tab, ) = st.tabs([
            "🏗️ Project Structure",
            "🏋️ Training Pipeline",
            "📊 Evaluation Metrics",
            "🗂️ Data & Cleaning",
            "🧠 Models",
            "🔬 Experiments & Results",
        ])



    # ============================================================
    # TAB — PROJECT STRUCTURE
    # ============================================================
    
    with structure_tab:
        st.header("📁 Project Structure")
        st.markdown(
            "TransistorAI is organized into independent components "
            "for dataset preparation, model development, evaluation "
            "and presentation."
        )

        # ----------------------------------------------------
        # SECTION 1 — REPOSITORY ORGANIZATION
        # ----------------------------------------------------
        st.subheader("Repository Organization")

        folders = [
            (
                "⚙️ configs/",
                "Experiment Configuration",
                "JSON files defining model selection, datasets, "
                "hyperparameters and experiment output paths."
            ),
            (
                "🗂️ data/",
                "Dataset Management",
                "German–English datasets, processed splits, "
                "training examples and human preference data."
            ),
            (
                "📝 docs/",
                "Documentation",
                "Project plans, methodology, experiment reports "
                "and presentation materials."
            ),
            (
                "📊 evaluation/",
                "Evaluation",
                "Scripts for BLEU, chrF, COMET, terminology checks, "
                "error analysis and human preference comparisons."
            ),
            (
                "🧠 models/",
                "Model Storage",
                "Locally stored MarianMT checkpoints and "
                "other model artifacts."
            ),
            (
                "📈 results/",
                "Experimental Results",
                "Training outputs, translation quality scores, "
                "model comparisons and analysis results."
            ),
            (
                "🧩 src/",
                "Reusable Functions",
                "Shared Python utilities for dataset loading, "
                "tokenization and truncation analysis."
            ),
        ]

        columns = st.columns(4, gap="small")

        # First row: 4 cards
        first_row = st.columns(4, gap="small")

        for index, (folder, title, description) in enumerate(folders[:4]):
            with first_row[index]:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <h3 style="text-align: center; font-size: 22px; margin-bottom: 12px;">
                            {folder}
                        </h3>
                        <p style="text-align: center; font-size: 14px; color: #A0A0A0;">
                            {title}
                        </p>
                        <p style="text-align: center; font-size: 15px; line-height: 1.6;">
                            {description}
                        </p>
                        """,
                        unsafe_allow_html=True,
                    )

        # Second row: 3 centered cards
        spacer_left, col1, col2, col3, spacer_right = st.columns(
            [0.5, 1, 1, 1, 0.5],
            gap="small"
        )

        for column, (folder, title, description) in zip(
            [col1, col2, col3],
            folders[4:]
        ):
            with column:
                with st.container(border=True):
                    st.markdown(
                        f"""
                        <h3 style="text-align: center; font-size: 22px; margin-bottom: 12px;">
                            {folder}
                        </h3>
                        <p style="text-align: center; font-size: 14px; color: #A0A0A0;">
                            {title}
                        </p>
                        <p style="text-align: center; font-size: 15px; line-height: 1.6;">
                            {description}
                        </p>
                        """,
                        unsafe_allow_html=True,
                    )

        st.divider()


        # ----------------------------------------------------
        # SECTION 2  — PROJECT WORKFLOW
        # ----------------------------------------------------
        st.subheader("How the Components Work Together")

        st.graphviz_chart("""
        digraph {
            rankdir=LR;
            graph [bgcolor="transparent", pad="0.2"];
            node [
                shape=box,
                style="rounded,filled",
                fillcolor="#243244",
                fontcolor="white",
                color="#536477",
                margin="0.18,0.12"
            ];
            edge [color="#8B9CAD"];

            Configs [label="configs/"];
            Data [label="data/"];
            Utils [label="src/"];
            Train [label="train.py"];
            Models [label="models/"];
            Eval [label="evaluation/"];
            Results [label="results/"];
            App [label="Streamlit MVP"];

            Configs -> Train;
            Data -> Utils;
            Utils -> Train;
            Train -> Models;
            Models -> Eval;
            Data -> Eval;
            Eval -> Results;
            Models -> App;
            Results -> App;
        }
        """, use_container_width=True)

        st.caption(
            "The workflow separates experiment configuration, "
            "data processing, training and evaluation. "
            "The Streamlit application brings the models and "
            "results together for interactive exploration."
        )

        st.divider()
        # ----------------------------------------------------
        # SECTION 3 — KEY PYTHON FILES
        # ----------------------------------------------------
        st.subheader("Key Python Files")


        python_files = [
            (
                "train.py",
                "Model Training",
                """
                **Purpose:** Main entry point for fine-tuning the
                MarianMT translation models (E1 and E2).

                **Main responsibilities:**
                - Reads experiment settings from a JSON configuration.
                - Validates dataset and output paths.
                - Loads the pretrained model and tokenizer.
                - Loads and tokenizes training and validation datasets.
                - Checks token truncation and prepares padded batches.
                - Configures the Hugging Face Seq2SeqTrainer.
                - Runs training and saves checkpoints, metrics and
                  the final trained model.

                **Project role:** Converts prepared translation data
                into fine-tuned German-to-English translation models.
                """
            ),
            (
            "src/training_utils.py",
            "Training Utilities",
            """
            **Purpose:** Provides reusable functions for preparing datasets
            and checking input lengths during model training.

            **Functions used by `src/train.py`:**
            - `load_translation_dataset()` — loads translation datasets.
            - `tokenize_dataset()` — tokenizes training and validation examples.
            - `truncation_report()` — identifies potential sequence truncation.

            **Project role:** Separates reusable data preparation and
            validation utilities from the main training script (`src/train.py`).
            """
            ),
            
        ]

        for filename, subtitle, explanation in python_files:
            with st.expander(f"📄 {filename} — {subtitle}"):
                st.markdown(explanation)


        

        
    # ============================================================
    # TAB 1 — TRAINING PIPELINE
    # ============================================================
    with training_tab:
        st.header("⚙️ Training Pipeline — train.py")

        st.markdown("""
        `train.py` is the main training entry point for the
        MarianMT fine-tuning experiments E1 and E2.

        It uses:
        - **Hugging Face Transformers** for model training.
        - **PyTorch** as the underlying deep learning framework.
        - **src/training_utils.py** for reusable data operations.
        - **JSON configuration files** for experiment settings.
        """)

        
        steps = [
            {
                "title": "1. Load and validate configuration",
                "purpose": "Select the correct training experiment.",
                "explanation": """
                The script requires a JSON configuration file through
                the `--config` command-line argument.

                It checks that the configuration and datasets exist,
                and prevents accidental overwriting of completed
                experiments or existing checkpoints.

                This is especially important because E1 and E2
                use different training datasets.
                """,
                "code": '''args = parse_arguments()

with args.config.open("r", encoding="utf-8") as file:
    config = json.load(file)

train_path = Path(config["train_path"])
validation_path = Path(config["validation_path"])
checkpoint_dir = Path(config["output_dir"])
run_dir = Path(config["run_dir"])''',
                "why": "Protects experiments and makes training configurable."
            },
            {
                "title": "2. Set random seed",
                "purpose": "Improve experiment reproducibility.",
                "explanation": """
                A random seed controls supported sources of
                randomness used during model training.

                Using the same seed helps make comparisons between
                experiments more consistent, although it does not
                guarantee identical results on every device.
                """,
                "code": '''set_seed(config["seed"])''',
                "why": "Reduces variation caused by random initialization and sampling."
            },
            {
                "title": "3. Load pretrained model and tokenizer",
                "purpose": "Start from an existing translation model.",
                "explanation": """
                The tokenizer converts text into tokens that the
                neural network can process.

                The pretrained sequence-to-sequence model provides
                existing translation capabilities that can be
                adapted through fine-tuning.

                For E1 and E2, the configured base model is MarianMT.
                """,
                "code": '''model_id = config["model_id"]

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)''',
                "why": "Fine-tuning reuses pretrained language knowledge instead of training from scratch."
            },
            {
                "title": "4. Load training and validation datasets",
                "purpose": "Prepare the German–English translation examples.",
                "explanation": """
                The script loads two datasets:

                - **Training:** Used to update model parameters.
                - **Validation:** Used to monitor model performance
                  during training.

                The loading operation is delegated to
                `src/training_utils.py`.
                """,
                "code": '''train_dataset = load_translation_dataset(
    config["train_path"],
    config["source_column"],
    config["target_column"],
)

validation_dataset = load_translation_dataset(
    config["validation_path"],
    config["source_column"],
    config["target_column"],
)''',
                "why": "Separates the examples used for learning from those used for validation."
            },
            {
                "title": "5. Analyze token truncation",
                "purpose": "Check whether sentences exceed model input limits.",
                "explanation": """
                Translation examples can contain more tokens than
                the configured maximum sequence length.

                The script checks potential truncation separately
                for German source sentences and English targets,
                in both training and validation datasets.

                This helps identify whether important information
                could be lost during tokenization.
                """,
                "code": '''report = truncation_report(
    dataset,
    tokenizer,
    column,
    max_length,
)

print(f"{split_name}/{column}: {report}")''',
                "why": "Makes possible information loss visible before training."
            },
            {
                "title": "6. Tokenize the datasets",
                "purpose": "Convert text into numerical model inputs.",
                "explanation": """
                Neural networks do not directly process text strings.

                Tokenization transforms the German source and English
                target sentences into token representations.

                Your reusable `tokenize_dataset()` function applies
                the tokenizer and configured source/target lengths.
                """,
                "code": '''tokenized_train = tokenize_dataset(
    train_dataset,
    tokenizer,
    config["max_source_length"],
    config["max_target_length"],
    config["source_column"],
    config["target_column"],
)''',
                "why": "Produces the numerical examples required by MarianMT."
            },
            {
                "title": "7. Configure dynamic padding",
                "purpose": "Create batches with compatible sequence lengths.",
                "explanation": """
                Sentences contain different numbers of tokens.

                Dynamic padding extends shorter sequences to the
                length required by the current batch.

                This can avoid unnecessary padding compared with
                padding every example to a fixed global length.
                """,
                "code": '''data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    padding=True,
)''',
                "why": "Makes variable-length examples usable in batches."
            },
            {
                "title": "8. Inspect a sample batch",
                "purpose": "Check the structure of the training tensors.",
                "explanation": """
                Before training, the script creates a small batch
                from the tokenized training dataset.

                It prints the shape of each tensor, allowing you
                to inspect the dimensions of the model inputs
                and target labels.

                The script also stops if the training dataset
                is empty.
                """,
                "code": '''sample_count = min(
    config["per_device_train_batch_size"],
    len(tokenized_train),
)

if sample_count == 0:
    raise ValueError("Training dataset is empty.")

sample_batch = data_collator([
    tokenized_train[i] for i in range(sample_count)
])

for name, tensor in sample_batch.items():
    print(f"{name}: {tuple(tensor.shape)}")''',
                "why": "Helps detect data preparation problems before expensive training."
            },
            {
                "title": "9. Define training hyperparameters",
                "purpose": "Control how the model learns.",
                "explanation": """
                `Seq2SeqTrainingArguments` defines the training
                process using values from the JSON configuration.

                Important settings include:

                - **Learning rate:** Size of optimizer updates.
                - **Batch size:** Examples processed per device batch.
                - **Gradient accumulation:** Number of batches
                  accumulated before an optimizer update.
                - **Epochs:** Passes through the training dataset.
                - **Evaluation strategy:** When validation runs.
                - **Save strategy:** When checkpoints are saved.
                """,
                "code": '''training_args = Seq2SeqTrainingArguments(
    output_dir=config["output_dir"],
    learning_rate=config["learning_rate"],
    optim=config["optimizer"],
    per_device_train_batch_size=config[
        "per_device_train_batch_size"
    ],
    gradient_accumulation_steps=config[
        "gradient_accumulation_steps"
    ],
    num_train_epochs=config["num_train_epochs"],
    eval_strategy=config["eval_strategy"],
    save_strategy=config["save_strategy"],
    seed=config["seed"],
)''',
                "why": "Defines the optimization and checkpointing behavior."
            },
            {
                "title": "10. Initialize Hugging Face Trainer",
                "purpose": "Connect the model, datasets and training settings.",
                "explanation": """
                `Seq2SeqTrainer` manages the training loop.

                It receives the model, training arguments,
                tokenized datasets, tokenizer and data collator.

                The trainer handles operations such as forward
                passes, gradient calculation, optimizer updates
                and configured validation.
                """,
                "code": '''trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    processing_class=tokenizer,
    data_collator=data_collator,
)''',
                "why": "Provides a standardized training workflow."
            },
            {
                "title": "11. Save experiment configuration",
                "purpose": "Record the settings used for training.",
                "explanation": """
                Before training starts, the script saves a copy
                of the selected JSON configuration in the
                checkpoint directory.

                This makes it easier to identify the parameters
                associated with a training run.
                """,
                "code": '''output_dir = Path(config["output_dir"])
output_dir.mkdir(parents=True, exist_ok=True)

with open(output_dir / "training_config.json", "w") as file:
    json.dump(config, file, indent=2)''',
                "why": "Supports experiment tracking and reproducibility."
            },
            {
                "title": "12. Train and export the model",
                "purpose": "Run fine-tuning and preserve the outputs.",
                "explanation": """
                `trainer.train()` starts fine-tuning.

                After training completes, the script saves:

                - The final model and tokenizer.
                - The selected training configuration.
                - Training history.
                - Aggregate training metrics.
                - Trainer metrics and state.

                These artifacts support later evaluation,
                comparison and model reuse.
                """,
                "code": '''train_result = trainer.train()

run_dir = Path(config["run_dir"])
run_dir.mkdir(parents=True, exist_ok=True)

model_dir = run_dir / "final_model"
trainer.save_model(str(model_dir))
tokenizer.save_pretrained(str(model_dir))

with (run_dir / "training_history.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(trainer.state.log_history, file, indent=2)

with (run_dir / "train_metrics.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(train_result.metrics, file, indent=2)''',
                "why": "Creates the trained model and records its training results."
            },
        ]

        st.subheader("Training Workflow")

        st.graphviz_chart("""
        digraph {
            rankdir=LR;
            graph [bgcolor="transparent"];
            node [
                shape=box,
                style="rounded,filled",
                fillcolor="#243244",
                fontcolor="white",
                color="#536477"
            ];
            edge [color="#8B9CAD"];

            Config [label="JSON Config"];
            Data [label="Datasets"];
            Token [label="Tokenization"];
            Train [label="Fine-tuning"];
            Save [label="Saved Model"];
            Eval [label="Evaluation"];

            Config -> Train;
            Data -> Token;
            Token -> Train;
            Train -> Save;
            Save -> Eval;
        }
        """, use_container_width=True)


        st.caption(
            "Select a step to understand its purpose, "
            "implementation and importance."
        )

        for step in steps:
            with st.expander(step["title"]):
                st.markdown(f"**Objective:** {step['purpose']}")
                st.markdown(step["explanation"])

                st.markdown("**Relevant Python code**")
                st.code(step["code"], language="python")

                st.info(f"Why it matters: {step['why']}")


        st.subheader("Training Workflow")

        for step in steps:
            with st.expander(step["title"]):
                st.markdown(f"**Objective:** {step['purpose']}")
                st.markdown(step["explanation"])

                st.markdown("**Relevant Python code**")
                st.code(step["code"], language="python")

                st.info(f"Why it matters: {step['why']}")

        st.divider()
        st.subheader("Training Configuration")

        st.markdown(
            "The training script reads parameters from the "
            "selected JSON file rather than hardcoding them."
        )

        config_files = sorted(
            (PROJECT_ROOT / "configs").glob("*.json")
        ) if (PROJECT_ROOT / "configs").exists() else []

        if config_files:
            selected_config = st.selectbox(
                "Inspect a training configuration",
                config_files,
                format_func=lambda path: path.name,
                key="explorer_training_config",
            )

            try:
                with selected_config.open(encoding="utf-8") as f:
                    selected_settings = json.load(f)
                st.json(selected_settings)
            except (OSError, ValueError) as exc:
                st.error(f"Cannot read configuration: {exc}")
        else:
            st.info("No JSON configuration files found in configs/.")

        st.divider()
        st.subheader("Training Script Source Code")

        train_file = PROJECT_ROOT / "src" / "train.py"

        if train_file.is_file():
            with st.expander("View complete train.py"):
                st.code(
                    train_file.read_text(encoding="utf-8"),
                    language="python",
                )
        else:
            st.warning("train.py was not found in the project root.")

        st.divider()
        st.subheader("Saved Training Artifacts")

        st.markdown("""
        After successful training, the script saves:

        - `final_model/` — exported model and tokenizer.
        - `training_config.json` — experiment configuration.
        - `training_history.json` — logged training history.
        - `train_metrics.json` — aggregate training metrics.

        Checkpoints and trainer state are saved under the
        configured output directory.
        """)

        st.caption(
            "The precise parameter values depend on the "
            "JSON configuration selected for each experiment."
        )



# ============================================================
# Tab 2 — Evaluation Metrics
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
# TAB 3 — DATASETS AND DATA PREPARATION
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


                # Reuse the verified dataset table
        chart_data = dataset_df.melt(
            id_vars="Split",
            value_vars=["Technical", "General"],
            var_name="Domain",
            value_name="Sentence pairs",
        )

        chart = (
            alt.Chart(chart_data)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Split:N",
                    title="Dataset split",
                    sort=["Train", "Validation", "Test"],
                ),
                y=alt.Y(
                    "Sentence pairs:Q",
                    title="Sentence pairs",
                    scale=alt.Scale(domain=[0, 150000]),
                ),
                color=alt.Color("Domain:N", title="Domain"),
                tooltip=[
                    "Split:N",
                    "Domain:N",
                    alt.Tooltip("Sentence pairs:Q", format=","),
                ],
            )
            .properties(height=350)
        )

        st.altair_chart(
            chart,
            use_container_width=True,
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
            (
                "1. Parallel corpus alignment",
                "Read German and English KDE4 files line by line. "
                "Reject files with unequal line counts."
            ),
            (
                "2. Text normalization",
                "Repair encoding issues with ftfy, normalize Unicode "
                "to NFC, and collapse unnecessary whitespace."
            ),
            (
                "3. Remove empty and duplicate pairs",
                "Remove pairs with empty source or target text "
                "and deduplicate exact source-target pairs."
            ),
            (
                "4. Sentence length filtering",
                "Retain pairs containing between 2 and 100 words "
                "in each language."
            ),
            (
                "5. Quality checks",
                "Detect KDE metadata, placeholder mismatches, "
                "identical translations and extreme length ratios."
            ),
            (
                "6. Quality filtering",
                "Exclude metadata contamination, placeholder "
                "mismatches and extreme length ratios. "
                "Retain identical translations unless another "
                "exclusion flag applies."
            ),
            (
                "7. Mixed-domain preparation",
                "For E2, combine general and sampled technical "
                "training data under a fixed 101,401-example budget, "
                "using random seed 42."
            ),
            (
                "8. Data leakage prevention",
                "Check for exact-pair overlap between training, "
                "validation and test sets."
            ),
        ]

        for title, description in pipeline:
            with st.expander(title):
                st.write(description)
                st.divider()
        st.subheader("Reproducible Dataset Splitting")

        st.markdown("""
        The KDE4 technical corpus was split using unique
        German source sentences rather than individual
        translation pairs.

        This prevents multiple translations of the same
        German sentence from appearing in different splits.

        **Target split proportions:**
        - Training: 80%
        - Validation: 10%
        - Test: 10%

        **Random seed:** 42

        Because splitting is performed on unique sources,
        the final sentence-pair percentages may differ
        slightly from the target proportions.
        """)

        split_df = pd.DataFrame({
            "Split": ["Training", "Validation", "Test"],
            "Target percentage": [80, 10, 10],
        })

        split_chart = (
            alt.Chart(split_df)
            .mark_bar()
            .encode(
                x=alt.X("Split:N", sort=None, title="Dataset split"),
                y=alt.Y(
                    "Target percentage:Q",
                    title="Target percentage",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                tooltip=[
                    alt.Tooltip("Split:N"),
                    alt.Tooltip("Target percentage:Q", format=".0f"),
                ],
            )
            .properties(height=280)
        )

        st.altair_chart(
            split_chart,
            use_container_width=True,
        )

        st.info(
            "Data leakage prevention: German source sentences "
            "are disjoint across the technical training, "
            "validation and test sets."
        )

        st.divider()
        st.subheader("E2 — Mixed-Domain Training")

        technical_count = 76400
        general_count = 25001

        mixed_df = pd.DataFrame({
            "Domain": ["Technical", "General"],
            "Sentence pairs": [technical_count, general_count],
        })

        st.dataframe(
            mixed_df,
            hide_index=True,
            use_container_width=True,
        )

        mixed_chart = (
            alt.Chart(mixed_df)
            .mark_bar()
            .encode(
                x=alt.X("Domain:N", sort=None, title="Training domain"),
                y=alt.Y(
                    "Sentence pairs:Q",
                    title="Training sentence pairs",
                ),
                tooltip=[
                    alt.Tooltip("Domain:N"),
                    alt.Tooltip("Sentence pairs:Q", format=","),
                ],
            )
            .properties(height=280)
        )

        st.altair_chart(
            mixed_chart,
            use_container_width=True,
        )

        st.caption(
            "E2 uses a fixed budget of 101,401 training pairs. "
            "All 25,001 general examples are included, with "
            "76,400 technical examples sampled without replacement. "
            "Counts assume the recorded general training size."
        )

    
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

        
        # =====================================================
        # FULL-TEST MODEL COMPARISON
        # =====================================================

        results_path = (
            PROJECT_ROOT
            / "results/day14_llm_baseline/full_model_comparison.csv"
        )

        st.subheader("Full-Test Evaluation — E0 to E3")

        if not results_path.exists():
            st.warning("Full-test evaluation results not found.")
        else:
            results_df = pd.read_csv(results_path)

            st.caption(
                "Technical: 12,623 sentences | "
                "General: 1,997 sentences. "
                "All four systems evaluated using BLEU, chrF and COMET."
            )

            metric = st.selectbox(
                "Choose evaluation metric",
                ["BLEU", "chrF", "COMET"],
                key="explorer_full_metric",
            )

                        # Full-test evaluation chart with fixed axes
            metric_max = {
                "BLEU": 50,
                "chrF": 75,
                "COMET": 1.0,
            }

            chart = (
                alt.Chart(results_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "model:N",
                        title="Translation system",
                        sort=[
                            "E0_baseline",
                            "E1_technical",
                            "E2_mixed",
                            "E3_Qwen",
                        ],
                        axis=alt.Axis(labelAngle=0),
                    ),
                    xOffset=alt.XOffset("domain:N"),
                    y=alt.Y(
                        f"{metric}:Q",
                        title=f"{metric} score",
                        scale=alt.Scale(
                            domain=[0, metric_max[metric]]
                        ),
                    ),
                    color=alt.Color(
                        "domain:N",
                        title="Domain",
                    ),
                    tooltip=[
                        alt.Tooltip("model:N", title="Model"),
                        alt.Tooltip("domain:N", title="Domain"),
                        alt.Tooltip(
                            f"{metric}:Q",
                            title=metric,
                            format=".4f",
                        ),
                    ],
                )
                .properties(height=350)
            )

            st.altair_chart(
                chart,
                use_container_width=True,
            )

            st.dataframe(
                results_df.round(4),
                hide_index=True,
                use_container_width=True,
            )

            st.subheader("Best Model by Domain")

            for domain in ["technical", "general"]:
                domain_df = results_df[
                    results_df["domain"] == domain
                ]

                winner = domain_df.loc[
                    domain_df["BLEU"].idxmax()
                ]

                st.success(
                    f"{domain.title()}: {winner['model']} "
                    f"(BLEU {winner['BLEU']:.2f}, "
                    f"chrF {winner['chrF']:.2f}, "
                    f"COMET {winner['COMET']:.4f})"
                )

        # =====================================================
        # DAY 14 PILOT — SEPARATE FROM FULL TEST
        # =====================================================

        pilot_path = (
            PROJECT_ROOT
            / "data/project_explorer/day14_pilot_metrics.csv"
        )

        with st.expander("Day 14 — 100-Sentence Pilot"):
            st.caption(
                "Pilot sample: 50 technical and 50 general sentences. "
                "These scores are not directly interchangeable "
                "with full-test scores."
            )

            if pilot_path.exists():
                pilot_df = pd.read_csv(pilot_path)

                st.dataframe(
                    pilot_df,
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.warning("Pilot results not found.")

        # =====================================================
        # KEY FINDINGS
        # =====================================================

        st.divider()
        st.subheader("Key Experimental Findings")

        with st.expander(
            "Finding 1 — Technical specialization",
            expanded=True,
        ):
            st.markdown("""
            E1 achieved the highest technical BLEU, chrF
            and COMET scores on the full test set.

            Fine-tuning on KDE4 technical data improved
            translation quality in the target domain.
            """)

        with st.expander(
            "Finding 2 — General-domain trade-off"
        ):
            st.markdown("""
            E0 achieved the highest general-domain scores.

            E1 performed substantially worse on general
            text, consistent with a domain-specialization
            trade-off.

            The results alone do not establish the exact
            cause of this performance reduction.
            """)

        with st.expander(
            "Finding 3 — Mixed-domain fine-tuning"
        ):
            st.markdown("""
            E2 approached E1's technical performance
            while retaining stronger general-domain
            performance than E1.

            Mixed-domain training offers a useful
            compromise between specialization and
            generalization in these experiments.
            """)

        with st.expander(
            "Finding 4 — NMT vs LLM"
        ):
            st.markdown("""
            E3 (Qwen) underperformed the specialized
            NMT systems on technical translation.

            Its general-domain COMET score was relatively
            close to E0, although BLEU and chrF were lower.

            This demonstrates why complementary
            evaluation metrics are useful.
            """)

        # =====================================================
        # HUMAN EVALUATION
        # =====================================================

        st.divider()
        st.subheader("Human Preferences vs Automatic Metrics")

        human_path = (
            PROJECT_ROOT
            / "results/day17_human_metrics/metric_agreement_summary.csv"
        )

        agreement_path = (
            PROJECT_ROOT
            / "results/day16_human_agreement/agreement_summary.csv"
        )

        if human_path.exists():
            human_df = pd.read_csv(human_path)

            overall = human_df[
                human_df["domain"] == "overall"
            ].copy()

            st.markdown("#### Day 17 — Metric Agreement")

            st.dataframe(
                overall[
                    [
                        "metric",
                        "votes",
                        "unique_sentences",
                        "strict_agreement_percent",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )

                        # Human vs automatic metric agreement
            agreement_chart = (
                alt.Chart(overall)
                .mark_bar(size=75)
                .encode(
                    x=alt.X(
                        "metric:N",
                        title="Evaluation metric",
                        sort=["bleu", "chrf", "comet"],
                        axis=alt.Axis(labelAngle=0),
                    ),
                    y=alt.Y(
                        "strict_agreement_percent:Q",
                        title="Strict agreement (%)",
                        scale=alt.Scale(domain=[0, 100]),
                    ),
                    tooltip=[
                        alt.Tooltip("metric:N", title="Metric"),
                        alt.Tooltip(
                            "strict_agreement_percent:Q",
                            title="Agreement",
                            format=".1f",
                        ),
                        alt.Tooltip(
                            "votes:Q",
                            title="Human votes",
                        ),
                    ],
                )
                .properties(height=300)
            )

            st.altair_chart(
                agreement_chart,
                use_container_width=True,
            )

        if agreement_path.exists():
            agreement_df = pd.read_csv(agreement_path)

            overall_agreement = agreement_df[
                agreement_df["domain"] == "overall"
            ]

            if not overall_agreement.empty:
                agreement = overall_agreement.iloc[0]

                st.metric(
                    "Day 16 — Inter-rater Exact Agreement",
                    f"{agreement['exact_agreement_percent']:.1f}%",
                )

                st.caption(
                    f"Based on "
                    f"{int(agreement['participant_pairs'])} "
                    "shared sentence comparisons."
                )

        st.info(
            "Human evaluation is exploratory. The preliminary "
            "Day 17 analysis includes only 25 votes across "
            "21 unique sentences. Repeated votes and ties "
            "must be considered when interpreting agreement."
        )

        # =====================================================
        # LIMITATIONS
        # =====================================================

        st.divider()
        st.subheader("Evaluation Limitations")

        st.markdown("""
        - BLEU and chrF depend on reference wording.
        - COMET is a learned metric and is not infallible.
        - Automatic scores do not replace human evaluation.
        - The human preference sample remains small.
        - Model architecture and inference settings differ.
        - Results are specific to the evaluated datasets.
        - Higher automatic scores do not guarantee better
          translations for every individual sentence.
        """)


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
                "BLEU": "BLEU",
                "chrF": "chrF",
                "COMET": "COMET",
            }

            selected_metric = st.selectbox(
                "Choose evaluation metric",
                list(metric_labels.keys()),
                key="explorer_metric_selection",
            )

            metric_column = metric_labels[selected_metric]

                        # Full-test evaluation chart with fixed axes
            metric_max = {
                "BLEU": 50,
                "chrF": 75,
                "COMET": 1.0,
            }

            chart = (
                alt.Chart(results_df)
                .mark_bar()
                .encode(
                    x=alt.X(
                        "model:N",
                        title="Translation system",
                        sort=[
                            "E0_baseline",
                            "E1_technical",
                            "E2_mixed",
                            "E3_Qwen",
                        ],
                        axis=alt.Axis(labelAngle=0),
                    ),
                    xOffset=alt.XOffset("domain:N"),
                    y=alt.Y(
                        f"{metric}:Q",
                        title=f"{metric} score",
                        scale=alt.Scale(
                            domain=[0, metric_max[metric]]
                        ),
                    ),
                    color=alt.Color(
                        "domain:N",
                        title="Domain",
                    ),
                    tooltip=[
                        alt.Tooltip("model:N", title="Model"),
                        alt.Tooltip("domain:N", title="Domain"),
                        alt.Tooltip(
                            f"{metric}:Q",
                            title=metric,
                            format=".4f",
                        ),
                    ],
                )
                .properties(height=350)
            )

            st.altair_chart(
                chart,
                use_container_width=True,
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
