import hashlib


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

def render_human_preference_page(user_id, username):
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
    # PARTICIPANT MANAGEMENT — USER PROFILE
    # ============================================================

    if not user_id:
        st.error("Please select a user profile first.")
        st.stop()

    participant_id = user_id

    st.success(
        f"Evaluation active for {username}. "
        "Your progress is saved automatically."
    )

    with st.expander("Evaluation settings"):
        st.write("**Current user:**", username)
        st.write("**Participant ID:**")
        st.code(participant_id, language=None)

        completed_ids = get_completed_ids(participant_id)

        st.write(
            f"**Saved evaluations:** {len(completed_ids)}"
        )




    
    # ============================================================
    # SHARED EVALUATION + REMAINING SENTENCES
    # ============================================================

    df = load_dataset()

    participant_id = user_id
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
