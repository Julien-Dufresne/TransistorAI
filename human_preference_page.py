import hashlib
import uuid
import re

import streamlit as st
from evaluation.day15_human_preference import participant_exists

from evaluation.day15_human_preference import (
    build_options,
    export_votes,
    get_completed_ids,
    load_dataset,
    save_vote,
)


def participant_sentence_order(df, participant_id):
    def order_key(evaluation_id):
        value = f"{participant_id}:{evaluation_id}:order"
        return hashlib.sha256(value.encode()).hexdigest()

    return sorted(
        df["evaluation_id"].tolist(),
        key=order_key,
    )


def render_human_preference_page():
    st.title("Human Preference Evaluation")

    st.write(
        "Compare German-to-English translations and select "
        "the version you think is most accurate and natural."
    )

    st.info(
        "The model identities are hidden. "
        "Your selections will be saved for research evaluation."
    )

    # ============================================================
    # PARTICIPANT MANAGEMENT — NEW OR RESUME
    # ============================================================

    if "human_active_participant" not in st.session_state:
        st.session_state.human_active_participant = None

    if "human_participant_mode" not in st.session_state:
        st.session_state.human_participant_mode = "New evaluation"

    if "human_new_id" not in st.session_state:
        st.session_state.human_new_id = None

    with st.expander("Participant settings", expanded=True):

        active_id = st.session_state.human_active_participant

        if active_id:
            st.success("Participant evaluation active")

            st.text_input(
                "Your Participant ID",
                value=active_id,
                disabled=True,
            )

            st.code(active_id, language=None)

            st.caption(
                "Save this ID to resume your evaluation "
                "in a future browser session."
            )

            if st.button("Switch participant"):
                st.session_state.human_active_participant = None
                st.session_state.human_new_id = None
                st.rerun()

        else:
            mode = st.radio(
                "What would you like to do?",
                ["New evaluation", "Resume evaluation"],
                key="human_participant_mode",
            )

            if mode == "New evaluation":

                if st.session_state.human_new_id is None:
                    st.session_state.human_new_id = uuid.uuid4().hex

                new_id = st.session_state.human_new_id

                st.write("Your new Participant ID:")
                st.code(new_id, language=None)

                st.info(
                    "Copy and save this ID before starting. "
                    "You will need it to resume later."
                )

                if st.button("Start evaluation", type="primary"):
                    st.session_state.human_active_participant = new_id
                    st.rerun()

            else:
                resume_id = st.text_input(
                    "Enter your existing Participant ID",
                    placeholder="Paste your 32-character ID",
                    key="human_resume_id",
                ).strip().lower()

                if st.button("Resume evaluation", type="primary"):

                    if not re.fullmatch(r"[0-9a-f]{32}", resume_id):
                        st.error(
                            "Invalid Participant ID. "
                            "Enter the 32-character hexadecimal ID."
                        )

                    elif not participant_exists(resume_id):
                        st.error(
                            "No saved votes were found for this ID. "
                            "Check the ID or start a new evaluation."
                        )

                    else:
                        st.session_state.human_active_participant = resume_id
                        st.rerun()

        participant_id = st.session_state.human_active_participant

        if not participant_id:
            st.info(
                "Start a new evaluation or resume an existing one "
                "to continue."
            )
            return

    # ============================================================
    # DATASET SELECTION
    # ============================================================

    st.subheader("Choose evaluation dataset")

    selected_domain = st.radio(
        "Which type of translations would you like to evaluate?",
        options=["general", "technical"],
        format_func=lambda domain: {
            "general": "🌍 General translations",
            "technical": "💻 Technical translations",
        }[domain],
        horizontal=True,
        key="preference_domain",
    )

    # Load the complete dataset.
    df = load_dataset()

    # Keep only sentences from the selected domain.
    df = df[
        df["domain"] == selected_domain
    ].reset_index(drop=True)

    # Retrieve all previously completed evaluations.
    completed_all = get_completed_ids(participant_id)

    # Count completed evaluations for the selected domain only.
    completed = completed_all.intersection(
        set(df["evaluation_id"])
    )

    total = len(df)

    st.progress(len(completed) / total)
    st.caption(f"{len(completed)} of {total} sentences evaluated")

    ordered_ids = participant_sentence_order(df, participant_id)

    remaining = [
        evaluation_id
        for evaluation_id in ordered_ids
        if evaluation_id not in completed
    ]

    if not remaining:
        st.success(
            f"You have completed all {total} "
            f"{selected_domain} evaluations!"
        )
        return

    evaluation_id = remaining[0]

    row = df.set_index("evaluation_id", drop=False).loc[evaluation_id]

    st.subheader("German source")
    st.write(row["source"])

    st.caption(
        f"Sentence {len(completed) + 1} of {total} "
        f"· Domain: {row['domain']}"
    )

    options = build_options(row, participant_id)

    st.subheader("Which English translation is best?")

    if len(options) == 1:
        st.info(
            "All four models generated the same translation. "
            "You can record this as a tie."
        )

    labels = [option["label"] for option in options]

    with st.form(key=f"vote_{participant_id}_{evaluation_id}"):
        selected_label = st.radio(
            "Choose one translation",
            options=labels,
            format_func=lambda label: next(
                option["text"]
                for option in options
                if option["label"] == label
            ),
            index=None,
        )

        submitted = st.form_submit_button(
            "Save vote and continue",
            type="primary",
        )

    if submitted:
        if selected_label is None:
            st.warning("Select a translation first.")
            return

        try:
            save_vote(
                participant_id,
                row,
                selected_label,
                options,
            )
        except Exception as exc:
            st.error(f"Could not save vote: {exc}")
            return

        st.rerun()

    with st.expander("Research data export"):
        st.warning(
            "This export contains votes from all participants. "
            "Only share it with authorized researchers."
        )

        st.download_button(
            "Download all votes (CSV)",
            data=export_votes().to_csv(index=False),
            file_name="human_preference_votes.csv",
            mime="text/csv",
        )
