
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

def render_vocabulary_page():
    """Render the Daily German Vocabulary Trainer."""

    st.header("📚 Daily German Vocabulary")
    st.caption(
        "Learn five new B1–B2 German words every day "
        "and review vocabulary from previous lessons."
    )

    initialize_database()

    # Keep the learner ID stable across browser sessions.
    # A fixed ID is suitable for this single-user local MVP.
    learner_id = "default_learner"

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

    # New vocabulary section.
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

    st.divider()

    # Review vocabulary section.
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

    for index, word in enumerate(daily_words, start=1):
        word_id = word["word_id"]
        german = word["german"]
        expected = word["english"]

        st.markdown(f"**{index}. {german}**")

        if word_id in attempts:
            attempt = attempts[word_id]

            if attempt["correct"]:
                st.success(
                    f"Correct! {german} → {expected}"
                )
            else:
                st.error(
                    f"Your answer: {attempt['answer']}\n\n"
                    f"Expected: {expected}"
                )

        else:
            with st.form(
                key=f"vocabulary_quiz_{learner_id}_{today}_{word_id}"
            ):
                answer = st.text_input(
                    "English translation",
                    key=f"answer_{learner_id}_{today}_{word_id}",
                )

                submitted = st.form_submit_button(
                    "Check answer"
                )

            if submitted:
                if not answer.strip():
                    st.warning("Please enter an answer.")
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
                        st.info("This word was already answered.")

        st.divider()

    if total and completed == total:
        st.success(
            f"🎉 Daily quiz completed! "
            f"Score: {correct}/{total}"
        )

