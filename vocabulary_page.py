
"""
Day 19 — Daily German Vocabulary Trainer.

Displays five new B1–B2 German words and up to five
review words from previous lessons.
"""

from datetime import date

import streamlit as st

from src.vocabulary_db import initialize_database
from src.vocabulary_scheduler import get_daily_words

from src.vocabulary_db import (
    submit_daily_answer,
    get_daily_attempts,
)
from src.vocabulary_quiz import check_answer

def render_vocabulary_page(learner_id):
    """Render the vocabulary trainer for the selected user."""

    st.header("📚 Daily German Vocabulary")
    st.caption(
        "Learn five new B1–B2 German words every day "
        "and review vocabulary from previous lessons."
    )

    initialize_database()

    # Keep the learner ID stable across browser sessions.
    # A fixed ID is suitable for this single-user local MVP.
    

    today = date.today().isoformat()

    daily_words = get_daily_words(
        learner_id=learner_id,
        study_date=today,
    )

    new_words = [
        word for word in daily_words
        if word["assignment_type"] == "new"
    ]

    review_words = [
        word for word in daily_words
        if word["assignment_type"] == "review"
    ]

    # Daily progress overview.
    col1, col2, col3 = st.columns(3)

    col1.metric("New words", len(new_words))
    col2.metric("Review words", len(review_words))
    col3.metric("Today's vocabulary", len(daily_words))

    st.divider()

    
    # ========================================================
    # DAILY VOCABULARY — SIDE-BY-SIDE LAYOUT
    # ========================================================

    new_col, review_col = st.columns(2, gap="large")

    # LEFT COLUMN — Today's new words
    with new_col:
        st.subheader("🆕 Today's New Words")

        if not new_words:
            st.success(
                "You have completed all available new vocabulary!"
            )
        else:
            for index, word in enumerate(new_words, start=1):
                with st.expander(
                    f"{index}. {word['german']} "
                    f"({word['level']})"
                ):
                    st.write(f"**English:** {word['english']}")
                    st.caption(
                        f"Category: {word['category'].capitalize()}"
                    )

    # RIGHT COLUMN — Words to review
    with review_col:
        st.subheader("🔁 Words to Review")

        if not review_words:
            st.info(
                "No review words yet. Come back tomorrow "
                "to review today's vocabulary."
            )
        else:
            for index, word in enumerate(review_words, start=1):
                with st.expander(
                    f"{index}. {word['german']} "
                    f"({word['level']})"
                ):
                    st.write(f"**English:** {word['english']}")
                    st.caption(
                        f"Category: {word['category'].capitalize()}"
                    )

    st.divider()



    
    # ========================================================
    # DAILY VOCABULARY QUIZ
    # ========================================================

    st.subheader("✍️ Daily Vocabulary Quiz")
    st.caption(
        "Translate the German expressions into English. "
        "Each word can be submitted once per day."
    )

    attempts = get_daily_attempts(learner_id, today)

    completed = sum(
        word["word_id"] in attempts
        for word in daily_words
    )

    correct = sum(
        attempts[word["word_id"]]["correct"]
        for word in daily_words
        if word["word_id"] in attempts
    )

    total = len(daily_words)

    if total:
        st.progress(
            completed / total,
            text=f"{completed}/{total} questions completed",
        )

    st.write(f"**Correct answers:** {correct}/{completed}")

    
    # ========================================================
    # DAILY VOCABULARY QUIZ — 5 QUESTIONS PER ROW
    # ========================================================

    columns_per_row = 5

    for row_start in range(0, len(daily_words), columns_per_row):

        row_words = daily_words[
            row_start:row_start + columns_per_row
        ]

        quiz_columns = st.columns(
            columns_per_row,
            gap="small",
        )

        for offset, word in enumerate(row_words):

            index = row_start + offset + 1

            word_id = word["word_id"]
            german = word["german"]
            expected = word["english"]

            with quiz_columns[offset]:

                with st.container(border=True):

                    st.markdown(
                        f"**{index}. {german}**"
                    )

                    if word_id in attempts:

                        attempt = attempts[word_id]

                        if attempt["correct"]:
                            st.success(
                                f"Correct! {expected}"
                            )
                        else:
                            st.error(
                                f"Your answer: {attempt['answer']}\n\n"
                                f"Expected: {expected}"
                            )

                    else:

                        with st.form(
                            key=(
                                f"vocabulary_quiz_"
                                f"{learner_id}_{today}_{word_id}"
                            )
                        ):

                            answer = st.text_input(
                                "English translation",
                                key=(
                                    f"answer_{learner_id}_"
                                    f"{today}_{word_id}"
                                ),
                            )

                            submitted = st.form_submit_button(
                                "Check answer"
                            )

                        if submitted:

                            if not answer.strip():
                                st.warning(
                                    "Please enter an answer."
                                )

                            else:

                                is_correct = check_answer(
                                    answer,
                                    expected,
                                )

                                saved = submit_daily_answer(
                                    learner_id=learner_id,
                                    word_id=word_id,
                                    user_answer=answer,
                                    is_correct=is_correct,
                                    study_date=today,
                                )

                                if saved:
                                    st.rerun()
                                else:
                                    st.info(
                                        "This word was already answered."
                                    )

        st.write("")


    if total and completed == total:
        st.success(
            f"🎉 Daily quiz completed! "
            f"Score: {correct}/{total}"
        )

