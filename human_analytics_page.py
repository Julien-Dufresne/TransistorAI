
"""Streamlit dashboard for TransistorAI human evaluation."""

import pandas as pd
import streamlit as st

from evaluation.day15_human_preference import load_dataset
from evaluation.day16_analyze_human_agreement import (
    load_votes,
    calculate_pairwise_agreement,
    agreement_summary,
    model_preferences,
    shared_subset_coverage,
)
from evaluation.day17_compare_human_metrics import (
    build_sentence_metrics,
    compare_human_metrics,
    summarize,
)


@st.cache_data(ttl=60, show_spinner="Loading evaluation analytics...")
def load_analytics():
    dataset = load_dataset()
    indexed = dataset.set_index("evaluation_id", drop=False)

    votes = load_votes()

    pairwise = calculate_pairwise_agreement(votes, indexed)
    agreement = agreement_summary(pairwise)
    preferences = model_preferences(votes)
    coverage = shared_subset_coverage(votes, indexed)

    # Sentence metrics are deterministic for the frozen dataset.
    scores = build_sentence_metrics(dataset)
    comparisons = compare_human_metrics(scores, votes)
    metric_summary = summarize(comparisons)

    return (
        dataset, votes, agreement,
        preferences, coverage, metric_summary,
    )


def render_human_analytics_page():
    st.title("📊 Human Evaluation Analytics")

    st.warning(
        "Demo mode: current votes were submitted randomly "
        "for testing. Agreement and preference statistics "
        "are illustrative, not research conclusions."
    )

    if st.button("🔄 Refresh analytics"):
        load_analytics.clear()
        st.rerun()

    try:
        (
            dataset, votes, agreement,
            preferences, coverage, metric_summary,
        ) = load_analytics()
    except Exception as exc:
        st.error(f"Unable to load analytics: {exc}")
        return

    domain = st.selectbox(
        "Evaluation domain",
        ["overall", "general", "technical"],
        format_func=lambda value: value.capitalize(),
    )

    filtered_votes = (
        votes if domain == "overall"
        else votes[votes["domain"] == domain]
    )

    filtered_dataset = (
        dataset if domain == "overall"
        else dataset[dataset["domain"] == domain]
    )

    st.subheader("Evaluation overview")

    a, b, c, d = st.columns(4)

    a.metric(
        "Participants",
        filtered_votes["participant_id"].nunique(),
    )
    b.metric("Votes", len(filtered_votes))
    c.metric(
        "Sentences rated",
        filtered_votes["evaluation_id"].nunique(),
    )
    d.metric(
        "Dataset size",
        len(filtered_dataset),
    )

    progress = (
        filtered_votes["evaluation_id"].nunique()
        / len(filtered_dataset)
        if len(filtered_dataset) else 0
    )

    st.progress(progress)
    st.caption(f"{progress:.1%} of sentences evaluated at least once")

    st.divider()

    st.subheader("Human agreement")

    current = agreement[agreement["domain"] == domain]

    if not current.empty and pd.notna(
        current.iloc[0]["exact_agreement_percent"]
    ):
        st.metric(
            "Exact translation agreement",
            f"{current.iloc[0]['exact_agreement_percent']:.1f}%",
        )
    else:
        st.info("At least two votes per sentence are needed.")

    agreement_chart = agreement[
        agreement["domain"].isin(["general", "technical"])
    ].copy()

    if domain != "overall":
        agreement_chart = agreement_chart[
            agreement_chart["domain"] == domain
        ]

    agreement_chart = agreement_chart.dropna(
        subset=["exact_agreement_percent"]
    )

    if not agreement_chart.empty:
        st.bar_chart(
            agreement_chart.set_index("domain")[
                "exact_agreement_percent"
            ],
            y_label="Agreement (%)",
        )

    st.caption(
        "Agreement is calculated across participant pairs "
        "who evaluated the same sentence."
    )

    st.subheader("Shared evaluation coverage")

    domain_coverage = (
        coverage if domain == "overall"
        else coverage[coverage["domain"] == domain]
    )

    coverage_counts = (
        domain_coverage["participant_count"]
        .value_counts()
        .sort_index()
        .rename_axis("Participants per sentence")
        .reset_index(name="Sentences")
    )

    st.dataframe(coverage_counts, hide_index=True)

    st.divider()

    st.subheader("Model preferences")

    st.caption(
        "Identical translations may represent multiple models. "
        "Fractional credit divides one vote equally among "
        "the models producing the selected translation."
    )

    
    # ============================================================
    # MODEL PREFERENCES — DOMAIN FILTERING AND OVERALL TOTALS
    # ============================================================

    # The preference analysis stores general and technical
    # results separately. For the overall view, combine both.
    if domain == "overall":
        selected_preferences = preferences
    else:
        selected_preferences = preferences[
            preferences["domain"] == domain
        ]

    # Initialize all four models with zero votes so the
    # dashboard remains stable even when no data is available.
    totals = pd.Series(
        0.0,
        index=["E0", "E1", "E2", "E3"],
        name="fractional_selections",
    )

    # Aggregate fractional selections across participants.
    # Each original vote contributes exactly one total unit,
    # even when multiple models produced identical text.
    if not selected_preferences.empty:
        totals = (
            selected_preferences.groupby("model")[
                "fractional_selections"
            ]
            .sum()
            .reindex(
                ["E0", "E1", "E2", "E3"],
                fill_value=0.0,
            )
        )


        
    # Descriptive labels for the four translation models.
    model_labels = {
        "E0": "E0 — Baseline NMT",
        "E1": "E1 — Technical NMT",
        "E2": "E2 — Mixed NMT",
        "E3": "E3 — Qwen LLM",
    }

    # Preserve model order and fractional vote calculations.
    display_totals = totals.rename(index=model_labels)

    # Chart with descriptive model names.
    st.bar_chart(
        display_totals,
        y_label="Fractional votes",
        x_label="Translation model",
        horizontal=True,
    )

    # Table with the same descriptive names.
    st.dataframe(
        display_totals.rename("Fractional votes")
        .rename_axis("Translation model")
        .reset_index(),
        hide_index=True,
        use_container_width=True,
    )


    st.divider()

    st.subheader("Human vs. automatic metrics")

    selected_metrics = metric_summary[
        metric_summary["domain"] == domain
    ].copy()

    if not selected_metrics.empty:
        selected_metrics["metric"] = (
            selected_metrics["metric"].str.upper()
        )

        chart = selected_metrics.set_index("metric")[
            [
                "strict_agreement_percent",
                "fractional_agreement_percent",
            ]
        ]

        chart.columns = [
            "Strict agreement",
            "Fractional agreement",
        ]

        st.bar_chart(
            chart,
            y_label="Human–metric agreement (%)",
        )

        st.dataframe(
            selected_metrics[
                [
                    "metric",
                    "votes",
                    "unique_sentences",
                    "strict_agreement_percent",
                    "fractional_agreement_percent",
                    "metric_tied_votes",
                ]
            ],
            hide_index=True,
        )

    st.caption(
        "Strict agreement credits a human choice if it matches "
        "any highest-scoring distinct translation. Fractional "
        "agreement divides credit across tied winners. "
        "BLEU and chrF are sentence-level scores."
    )

    st.divider()

    st.subheader("Methodology")

    with st.expander("How are these results calculated?"):
        st.markdown(
            """
            - **Human agreement:** Exact selected-text agreement
              across participant pairs evaluating the same sentence.
            - **Model preferences:** One vote is split equally
              between models producing identical selected text.
            - **BLEU:** Sentence-level BLEU with effective order.
            - **chrF:** Sentence-level character F-score.
            - **COMET:** Existing validated segment scores.
            - **Metric agreement:** Whether the human-selected
              translation is among the highest-scoring distinct
              translations for that sentence.
            - **Limitations:** Small sample sizes, repeated votes
              on the same sentences, and demo votes submitted randomly.
            """
        )
