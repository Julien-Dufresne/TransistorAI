
"""TransistorAI — My Mistakes learning dashboard."""

from collections import Counter

import streamlit as st

from evaluation.mistake_tracker import (
    get_user_mistakes,
    update_mistake_status,
)



def render_my_mistakes_page(user_id):
    st.subheader("🧠 My Mistakes")
    st.caption(
        "Review the mistakes you saved during Test Yourself "
        "and track your learning progress."
    )

    mistakes = get_user_mistakes(user_id)

    if not mistakes:
        st.info(
            "You haven't saved any mistakes yet. "
            "Complete a Test Yourself exercise and select "
            "mistakes to track them here."
        )
        return

   
    # --------------------------------------------------
    # Overview
    # --------------------------------------------------
    total = len(mistakes)
    mastered = sum(
        m["status"] == "mastered" for m in mistakes
    )
    reviewing = sum(
        m["status"] == "reviewing" for m in mistakes
    )
    to_review = total - mastered

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total mistakes", total)
    c2.metric("To review", to_review)
    c3.metric("Reviewing", reviewing)
    c4.metric("Mastered", mastered)

    st.progress(mastered / total)
    st.caption(
        f"{mastered / total:.0%} of saved mistakes mastered"
    )

    st.divider()

    # --------------------------------------------------
    # Category overview
    # --------------------------------------------------
    st.markdown("#### Mistakes by category")

    category_counts = Counter(
        m["category"] for m in mistakes
    )

    chart_data = {
        category.replace("_", " ").title(): count
        for category, count in category_counts.items()
    }

    st.bar_chart(
        chart_data,
        x_label="Mistake category",
        y_label="Number of mistakes",
        height=220,
        use_container_width=True,
    )

    st.divider()

    # --------------------------------------------------
    # Filters
    # --------------------------------------------------
    st.markdown("#### Your mistake history")

    categories = sorted(set(
        m["category"] for m in mistakes
    ))

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        selected_category = st.selectbox(
            "Category",
            ["All"] + categories,
            format_func=lambda value: (
                value.replace("_", " ").title()
            ),
            key="mistakes_category_filter",
        )

    with filter_col2:
        selected_status = st.selectbox(
            "Status",
            ["All", "new", "reviewing", "mastered"],
            format_func=lambda value: value.title(),
            key="mistakes_status_filter",
        )

    filtered = [
        m for m in mistakes
        if (
            selected_category == "All"
            or m["category"] == selected_category
        )
        and (
            selected_status == "All"
            or m["status"] == selected_status
        )
    ]

    st.caption(f"Showing {len(filtered)} mistake(s)")

    # --------------------------------------------------
    # Mistake cards
    # --------------------------------------------------
    for mistake in filtered:
        mistake_id = mistake["mistake_id"]
        category = mistake["category"].replace(
            "_", " "
        ).title()

        with st.container(border=True):
            header_left, header_right = st.columns(
                [3, 1]
            )

            with header_left:
                st.markdown(f"**{category}**")

            with header_right:
                st.caption(
                    mistake["status"].title()
                )

            left, right = st.columns(2)

            with left:
                st.markdown("**Your translation**")
                st.write(
                    mistake["original_text"] or "—"
                )

            with right:
                st.markdown("**Correction**")
                st.write(
                    mistake["corrected_text"] or "—"
                )

            if mistake["explanation"]:
                with st.expander("Explanation"):
                    st.write(mistake["explanation"])

            st.caption(
                f"Saved: {mistake['created_at'][:10]}"
            )

            status_options = [
                "new", "reviewing", "mastered"
            ]

            selected = st.selectbox(
                "Learning status",
                status_options,
                index=status_options.index(
                    mistake["status"]
                ),
                format_func=lambda value: value.title(),
                key=f"status_{user_id}_{mistake_id}",
            )

            if selected != mistake["status"]:
                update_mistake_status(
                    mistake_id=mistake_id,
                    user_id=user_id,
                    status=selected,
                )
                st.rerun()
