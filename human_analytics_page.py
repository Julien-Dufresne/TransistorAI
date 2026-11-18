
"""Streamlit dashboard for TransistorAI human evaluation."""
import json
import altair as alt
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
from pathlib import Path

DAY22_RESULTS = Path("results/day22_human_metrics")

@st.cache_data(ttl=60, show_spinner="Loading evaluation analytics...")
def load_analytics():
    dataset = load_dataset()
    indexed = dataset.set_index("evaluation_id", drop=False)

    votes = load_votes()

    pairwise = calculate_pairwise_agreement(votes, indexed)
    agreement = agreement_summary(pairwise)
    preferences = model_preferences(votes)
    coverage = shared_subset_coverage(votes, indexed)

    # Frozen Day 22 evaluation snapshot.
    # Live voting statistics remain independent.
    scores = pd.read_csv(
        DAY22_RESULTS / "sentence_level_metrics.csv"
    )

    comparisons = pd.read_csv(
        DAY22_RESULTS / "human_metric_comparison.csv"
    )

    metric_summary = pd.read_csv(
        DAY22_RESULTS / "metric_agreement_summary.csv"
    )

    return (
        dataset, votes, agreement,
        preferences, coverage, metric_summary, scores,
    )


def render_human_analytics_page():
    st.title("📊 Human Evaluation Analytics")

    # Disable mouse-wheel zoom on all Altair/Vega-Lite charts.

    st.info(
    "Final human evaluation: 200 sentences evaluated, "
    "204 saved votes from 3 participants. "
    "Results are exploratory because one participant "
    "contributed 200 of the 204 votes."
)

    if st.button("🔄 Refresh analytics"):
        load_analytics.clear()
        st.rerun()

    try:
        (
        dataset, votes, agreement,
        preferences, coverage, metric_summary, scores,
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
         "Exclusive model selections from the currently saved votes. "
        "Tied selections are excluded. The chart updates when "
        "new votes are saved."
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


    # ============================================================
    # DAY 22 — EXCLUSIVE MODEL PREFERENCES BY DOMAIN
    # ============================================================

    st.markdown("#### Exclusive model preferences by domain")

    exclusive_records = []

    for _, vote in votes.iterrows():
        selected = json.loads(vote["selected_models"])

        if len(selected) == 1:
            exclusive_records.append({
                "domain": vote["domain"],
                "model": selected[0],
            })

    exclusive_df = pd.DataFrame(
        exclusive_records,
        columns=["domain", "model"],
    )

    if not exclusive_df.empty:
        exclusive_counts = (
            exclusive_df.groupby(["model", "domain"])
            .size()
            .unstack(fill_value=0)
            .reindex(["E0", "E1", "E2", "E3"], fill_value=0)
            .reindex(
                columns=["general", "technical"],
                fill_value=0,
            )
        )

        st.bar_chart(
            exclusive_counts,
            x_label="Translation model",
            y_label="Exclusive human selections",
            stack=False,
        )

        st.dataframe(
            exclusive_counts.rename(
                columns={
                    "general": "General",
                    "technical": "Technical",
                }
            ),
            use_container_width=True,
        )

        st.caption(
            "E3 (Qwen) received the most exclusive human "
            "selections in both domains in the completed "
            "Day 22 evaluation. Tied selections are excluded."
        )


    st.divider()

    # ============================================================
    # DAY 22 — MODEL PERFORMANCE
    # ============================================================

    st.subheader("Model performance by domain")

    st.caption(
        "Final Day 22 evaluation: 200 unique sentences "
        "(100 general, 100 technical), with 800 model–sentence "
        "scores. Values are means of sentence-level metrics, "
        "not corpus-level scores."
    )

    performance = scores.copy()

    if domain != "overall":
        performance = performance[
            performance["domain"] == domain
        ]

    metric_choice = st.selectbox(
        "Automatic evaluation metric",
        ["BLEU", "chrF", "COMET"],
        key="day22_performance_metric",
    )

    metric_column = {
        "BLEU": "bleu",
        "chrF": "chrf",
        "COMET": "comet",
    }[metric_choice]

    performance_chart = performance.pivot_table(
        index="model",
        columns="domain",
        values=metric_column,
        aggfunc="mean",
    ).reindex(["E0", "E1", "E2", "E3"])

    chart_data = (
        performance_chart
        .reset_index()
        .melt(
            id_vars="model",
            var_name="domain",
            value_name="score",
        )
    )

    chart = (
        alt.Chart(chart_data)
        .mark_bar()
        .encode(
            x=alt.X("model:N", title="Translation model"),
            xOffset="domain:N",
            y=alt.Y("score:Q", title=f"Mean sentence-level {metric_choice}"),
            color="domain:N",
        )
    )

    st.altair_chart(chart, use_container_width=True)

    st.dataframe(
        performance_chart.round(4),
        use_container_width=True,
    )

    st.divider()

    st.caption(
    "Frozen Day 22 evaluation snapshot: 204 votes, "
    "200 unique sentences, and 800 model–sentence scores. "
    "Live votes may change, but these metric comparisons "
    "remain fixed until the analysis is recomputed."
)

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
            stack=False,
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
         "Final Day 22 evaluation snapshot: 204 votes from "
        "3 participants across 200 unique sentences. "
        "One participant contributed 200 votes."
    )

    st.divider()

    # ============================================================
    # DAY 22 — HUMAN/METRIC AGREEMENT PATTERNS
    # ============================================================

    st.subheader("Human–metric agreement patterns")

    st.caption(
         "Final Day 22 evaluation: 204 human votes across "
        "200 unique sentences. Each vote is counted once "
        "per metric."
    )

    pattern_data = pd.read_csv(
        DAY22_RESULTS / "human_metric_comparison.csv"
    )

    if domain != "overall":
        pattern_data = pattern_data[
            pattern_data["domain"] == domain
        ]

    if not pattern_data.empty:
        patterns = pattern_data.pivot(
            index=["evaluation_id", "participant_id", "domain"],
            columns="metric",
            values="strict_agreement",
        ).reset_index()

        def classify_pattern(row):
            matched = [
                metric.upper()
                for metric in ["bleu", "chrf", "comet"]
                if row[metric] == 1
            ]
            return " + ".join(matched) if matched else "None"

        patterns["pattern"] = patterns.apply(
            classify_pattern,
            axis=1,
        )

        counts = (
            patterns["pattern"]
            .value_counts()
            .rename_axis("Agreement pattern")
            .reset_index(name="Votes")
        )

        st.bar_chart(
            counts.set_index("Agreement pattern"),
            horizontal=True,
            x_label="Number of human votes",
            y_label="Agreement pattern",
        )

        st.dataframe(
            counts,
            hide_index=True,
            use_container_width=True,
        )

        none_count = int(
            (patterns["pattern"] == "None").sum()
        )

        total = len(patterns)

        st.metric(
            "Votes where no metric agreed",
            f"{none_count}/{total}",
            f"{none_count / total:.1%}" if total else None,
            delta_color="off",
        )

    st.divider()

        # ============================================================
        # DAY 20 — INTERACTIVE TRANSLATION CASE STUDIES
        # ============================================================

    st.subheader("Interactive Translation Case Studies")

    st.caption(
        "Explore all 200 sentences from the completed Day 22 "
        "human evaluation. Compare E0–E3 translations, "
        "BLEU, chrF, COMET, and human preferences."
    )

    # Use the frozen Day 19 evaluation snapshot.
    case_scores = scores.copy()

    if domain != "overall":
        case_scores = case_scores[
            case_scores["domain"] == domain
        ]

    available_ids = sorted(
        case_scores["evaluation_id"].unique()
    )

    if available_ids:

        selected_id = st.selectbox(
            "Select evaluation sentence",
            available_ids,
            key="day20_case_study_id",
        )

        case = case_scores[
            case_scores["evaluation_id"] == selected_id
        ].copy()

        case = (
            case.set_index("model")
            .reindex(["E0", "E1", "E2", "E3"])
            .reset_index()
        )

        first = case.iloc[0]

        st.markdown(
            f"**Domain:** {first['domain'].capitalize()}"
        )

        st.markdown("**German source**")
        st.info(first["source"])

        st.markdown("**English reference**")
        st.info(first["reference"])

        st.markdown("#### Model translations")

        display = case[
            ["model", "translation", "bleu", "chrf", "comet"]
        ].copy()

        display.columns = [
            "Model", "Translation", "BLEU", "chrF", "COMET"
        ]

        st.dataframe(
            display,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Model": st.column_config.TextColumn(
                    "Model", width="small"
                ),
                "Translation": st.column_config.TextColumn(
                    "Translation", width="large"
                ),
                "BLEU": st.column_config.NumberColumn(
                    "BLEU", format="%.2f", width="small"
                ),
                "chrF": st.column_config.NumberColumn(
                    "chrF", format="%.2f", width="small"
                ),
                "COMET": st.column_config.NumberColumn(
                    "COMET", format="%.4f", width="small"
                ),
            },
        )

        # Automatic metric winners.
        st.markdown("#### Automatic metric winners")

        winner_cols = st.columns(3)

        for col, metric, label in zip(
            winner_cols,
            ["bleu", "chrf", "comet"],
            ["BLEU", "chrF", "COMET"],
        ):
            best_score = case[metric].max()

            winners = case.loc[
                case[metric] == best_score, "model"
            ].tolist()

            col.metric(
                label,
                ", ".join(winners),
            )

        # Human preferences from the frozen Day 19 snapshot.
        comparisons = pd.read_csv(
            DAY22_RESULTS / "human_metric_comparison.csv"
        )

        human_case = comparisons[
            (comparisons["evaluation_id"] == selected_id)
            & (comparisons["metric"] == "bleu")
        ]

        st.markdown("#### Human preferences")

        if not human_case.empty:

            st.caption(
                f"{len(human_case)} human vote(s) "
                "recorded for this sentence."
            )

            for vote_number, (_, vote) in enumerate(
                human_case.iterrows(), start=1
            ):
                st.markdown(
                    f"**Vote {vote_number}:** "
                    f"{vote['human_selected_models']}"
                )

                st.write(
                    vote["human_selected_text"]
                )

        else:
            st.info(
                "No human preference recorded "
                "for this sentence."
            )

        # Display metric scores as a comparison chart.
        st.markdown("#### Metric comparison")

        chart_metric = st.selectbox(
            "Select metric to compare",
            ["BLEU", "chrF", "COMET"],
            key="day20_case_metric",
        )

        metric_column = {
            "BLEU": "bleu",
            "chrF": "chrf",
            "COMET": "comet",
        }[chart_metric]

        st.bar_chart(
            case.set_index("model")[metric_column],
            x_label="Translation model",
            y_label=chart_metric,
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
            - **Limitations:** One participant contributed 200 of the
            204 votes, so the results primarily reflect a single
            evaluator's preferences. Only four additional votes
            were collected from two other participants.
            - **Evaluation snapshot:** Automatic metric comparisons
            use the frozen Day 22 results. Live voting statistics
            may differ if additional votes are submitted.
            """
        )
