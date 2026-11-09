import hashlib
import uuid
import re

import streamlit as st

from evaluation.day15_human_preference import (
    build_options,
    export_votes,
    get_completed_ids,
    load_dataset,
    participant_exists,  # Check whether an ID has existing saved votes.
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


# Fixed shared subset for inter-annotator agreement.
# Keep this selection unchanged once real evaluation begins.

SHARED_SENTENCES_PER_DOMAIN = 10
SHARED_SUBSET_SEED = "transistorai-shared-v1"


def get_shared_evaluation_ids(df):
    shared_ids = []

    for domain in ["general", "technical"]:
        domain_df = df[df["domain"] == domain]

        ordered = sorted(
            domain_df["evaluation_id"].tolist(),
            key=lambda evaluation_id: hashlib.sha256(
                f"{SHARED_SUBSET_SEED}:{evaluation_id}".encode()
            ).hexdigest(),
        )

        if len(ordered) < SHARED_SENTENCES_PER_DOMAIN:
            raise ValueError(
                f"Insufficient sentences in {domain}"
            )

        shared_ids.extend(
            ordered[:SHARED_SENTENCES_PER_DOMAIN]
        )

    return shared_ids

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
    # PARTICIPANT MANAGEMENT — NEW AND RETURNING PARTICIPANTS
    # ============================================================

    # A participant ID is a UUID represented by 32 hexadecimal characters.
    # Normalize user input to allow uppercase IDs and surrounding spaces.
    def normalize_participant_id(value):
        return str(value).strip().lower()

    def is_valid_participant_id(value):
        return re.fullmatch(r"[0-9a-f]{32}", value) is not None

    # Initialize the active participant only once per Streamlit session.
    if "human_active_participant" not in st.session_state:
        st.session_state.human_active_participant = None

    # Restore a participant from the URL when possible.
    url_participant = normalize_participant_id(
        st.query_params.get("participant", "")
    )

    if is_valid_participant_id(url_participant):
        # Only restore IDs that already have saved votes.
        if participant_exists(url_participant):
            st.session_state.human_active_participant = url_participant

        # A newly created participant may not have voted yet.
        elif (
            st.session_state.human_active_participant
            != url_participant
        ):
            st.query_params.pop("participant", None)

    participant_id = st.session_state.human_active_participant

    # ------------------------------------------------------------
    # NEW PARTICIPANT / MANUAL RESUME
    # ------------------------------------------------------------

    if not participant_id:
        st.info(
            "Start a new anonymous evaluation or resume "
            "using your existing Participant ID."
        )

        if st.button("Start new evaluation", type="primary"):
            # Generate a unique participant identifier.
            participant_id = uuid.uuid4().hex

            st.session_state.human_active_participant = participant_id
            st.query_params["participant"] = participant_id

            st.rerun()

        st.divider()

        st.subheader("Resume an existing evaluation")

        # Use a form to avoid triggering validation on every keystroke.
        with st.form("resume_participant_form"):
            entered_id = st.text_input(
                "Participant ID",
                placeholder="Enter your 32-character Participant ID",
            )

            resume_submitted = st.form_submit_button(
                "Resume evaluation"
            )

        if resume_submitted:
            entered_id = normalize_participant_id(entered_id)

            if not is_valid_participant_id(entered_id):
                st.error(
                    "Invalid Participant ID format. "
                    "Enter a 32-character hexadecimal ID."
                )

            elif not participant_exists(entered_id):
                st.error(
                    "No saved evaluation was found for this ID. "
                    "Check the ID and try again."
                )

            else:
                # Restore the participant and synchronize the URL.
                st.session_state.human_active_participant = entered_id
                st.query_params["participant"] = entered_id

                st.rerun()

        return

    # ------------------------------------------------------------
    # ACTIVE PARTICIPANT
    # ------------------------------------------------------------

    # Synchronize the URL with the current participant.
    if st.query_params.get("participant") != participant_id:
        st.query_params["participant"] = participant_id

    st.success(
        "Your evaluation is active. "
        "Your progress is saved automatically."
    )

    with st.expander("Evaluation settings", expanded=False):
        st.write("**Your Participant ID**")
        st.code(participant_id, language=None)

        st.caption(
            "Save this ID or bookmark the current URL "
            "to resume your evaluation later."
        )

        # Display the number of previously completed evaluations.
        completed_ids = get_completed_ids(participant_id)

        st.write(
            f"**Saved evaluations:** {len(completed_ids)}"
        )

        st.divider()

        if st.button("Switch participant"):
            # Clear the active participant without deleting votes.
            st.session_state.human_active_participant = None
            st.query_params.pop("participant", None)
            st.rerun()




    
    # ============================================================
    # SHARED EVALUATION + REMAINING SENTENCES
    # ============================================================

    df = load_dataset()

    participant_id = st.session_state.human_active_participant
    completed_all = get_completed_ids(participant_id)

    shared_ids = get_shared_evaluation_ids(df)
    shared_set = set(shared_ids)

    completed_shared = shared_set.intersection(completed_all)

    st.subheader("Evaluation progress")

    st.progress(len(completed_all) / len(df))
    st.caption(
        f"{len(completed_all)} of {len(df)} sentences evaluated"
    )

    if len(completed_shared) < len(shared_ids):
        st.info(
            "Shared evaluation phase: "
            f"{len(completed_shared)} of {len(shared_ids)} "
            "sentences completed."
        )

        remaining = [
            evaluation_id
            for evaluation_id in shared_ids
            if evaluation_id not in completed_all
        ]

    else:
        st.success("Shared evaluation completed!")

        remaining_ids = [
            evaluation_id
            for evaluation_id in participant_sentence_order(
                df, participant_id
            )
            if evaluation_id not in shared_set
            and evaluation_id not in completed_all
        ]

        remaining = remaining_ids

    if not remaining:
        st.success("You have completed all evaluations!")
        return

    evaluation_id = remaining[0]

    row = df.set_index(
        "evaluation_id", drop=False
    ).loc[evaluation_id]

    total = len(df)

   


    st.subheader("German source")
    st.write(row["source"])

    st.caption(
        f"Sentence {len(completed_all) + 1} of {total} "
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
