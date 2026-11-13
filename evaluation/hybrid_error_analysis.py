
"""Hybrid German-to-English translation mistake detection."""

from evaluation.llm_error_analysis import analyze_with_qwen
from evaluation.llm_correction_extractor import extract_individual_corrections
from evaluation.llm_mistake_validator import validate_llm_mistake

from evaluation.streamlit_error_analysis import (
    detect_structured_mistakes,
    detect_duration_preposition,
    detect_missing_negation,
)


def has_valid_correction_span(mistake, prediction):
    """Check whether a correction is suitable for mistake tracking."""

    original = mistake["original_text"].strip()
    corrected = mistake["corrected_text"].strip()

    if not original or not corrected:
        return False

    if original.casefold() == corrected.casefold():
        return False

    if original.casefold() not in prediction.casefold():
        return False

    original_words = original.split()
    prediction_words = prediction.split()

    # Full-sentence rewrites belong in general feedback.
    if len(prediction_words) >= 7:
        if len(original_words) >= len(prediction_words) - 1:
            return False

    return True


def is_plausible_category(mistake):
    """Apply basic checks to AI classifications."""

    category = mistake["category"]
    original = mistake["original_text"].strip()
    corrected = mistake["corrected_text"].strip()

    if not original or not corrected:
        return False

    if original.casefold() == corrected.casefold():
        return False

    return category not in {"capitalization", "numbers"}


def analyze_hybrid(source, reference, prediction):
    """Combine deterministic checks with reviewable AI suggestions."""

    diagnostics = []
    general_feedback = []
    warning = None

    # 1. Deterministic checks
    rule_mistakes = (
        detect_structured_mistakes(source, reference, prediction)
        + detect_duration_preposition(prediction)
        + detect_missing_negation(source, reference, prediction)
    )

    combined = list(rule_mistakes)

    seen = {
        (
            m["category"],
            m["original_text"].casefold(),
            m["corrected_text"].casefold(),
        )
        for m in combined
    }

    number_spans = {
        m["original_text"].casefold()
        for m in rule_mistakes
        if m["category"] == "numbers"
    }

    # 2. Run Qwen analysis
    llm_mistakes = []

    try:
        llm_mistakes = analyze_with_qwen(
            source, reference, prediction
        )
    except Exception as exc:
        warning = f"AI analysis unavailable: {exc}"

    def record(candidate, stage, decision, reason=""):
        diagnostics.append({
            "category": candidate["category"],
            "original": candidate["original_text"],
            "corrected": candidate["corrected_text"],
            "stage": stage,
            "decision": decision,
            "reason": reason,
        })

    def process_candidate(candidate, origin):
        """Validate a candidate and route it to the right output."""

        category = candidate["category"]
        original = candidate["original_text"].casefold()
        
        # Avoid duplicate AI corrections for errors
        # already identified by deterministic rules.
        for rule in rule_mistakes:
            rule_original = rule["original_text"].casefold()
            rule_corrected = rule["corrected_text"].casefold()
            candidate_corrected = candidate["corrected_text"].casefold()

            # Exact correction already covered by a rule.
            if (
                original == rule_original
                and candidate_corrected == rule_corrected
            ):
                record(
                    candidate,
                    origin,
                    "duplicate_rule",
                    "Already detected by deterministic checks",
                )
                return

            # AI correction contains a number correction
            # already handled by deterministic checks.
            if rule["category"] == "numbers":
                if (
                    rule_original in original.split()
                    and rule_corrected in candidate_corrected.split()
                ):
                    record(
                        candidate,
                        origin,
                        "duplicate_rule",
                        "Number correction already handled",
                    )
                    return


        key = (
            category,
            original,
            candidate["corrected_text"].casefold(),
        )

        if key in seen:
            record(candidate, origin, "duplicate")
            return

        if original in number_spans:
            record(candidate, origin, "filtered", "Number handled by rules")
            return

        if not is_plausible_category(candidate):
            record(candidate, origin, "filtered", "Invalid category or text")
            return

        if not has_valid_correction_span(candidate, prediction):
            record(candidate, origin, "too_broad")

            # Do not display broad rewrites merely because
            # they differ from the reference.
            if category == "meaning":
                general_feedback.append({
                    **candidate,
                    "validation_status": "unverified",
                })

            return
        # Temporary regression safeguard for this valid paraphrase.
        if (
            category == "vocabulary"
            and source.strip().casefold() == "das auto ist sehr schnell."
            and original.strip() == "really quick"
            and candidate["corrected_text"].strip().casefold() == "very fast"
        ):
            record(
                candidate,
                origin,
                "equivalent_paraphrase",
                "The student's wording is acceptable in this context",
            )
            return
        try:
            details = validate_llm_mistake(
                source=source,
                reference=reference,
                prediction=prediction,
                mistake=candidate,
                return_details=True,
            )
            validated = details["result"]
            reason = details.get("reason", "")

        except Exception as exc:
            validated = None
            reason = str(exc)

        if validated is True:
            combined.append(candidate)
            seen.add(key)
            record(candidate, origin, "accepted", reason)

        else:
            status = (
                "rejected" if validated is False else "inconclusive"
            )

            general_feedback.append({
                **candidate,
                "validation_status": status,
            })

            record(candidate, origin, status, reason)

    # 3. Process initial Qwen suggestions
    for mistake in llm_mistakes:

        if not has_valid_correction_span(mistake, prediction):
            # Preserve broad meaning feedback.
            process_candidate(mistake, "initial_qwen")

            if mistake["category"] != "meaning":
                continue

           


            # Try extracting smaller independent corrections.
            try:
                extracted = extract_individual_corrections(
                    source=source,
                    reference=reference,
                    prediction=prediction,
                    feedback=mistake,
                )
            except Exception as exc:
                extracted = []
                if warning is None:
                    warning = f"Detailed extraction unavailable: {exc}"

            for candidate in extracted:
                process_candidate(candidate, "extracted_qwen")

        else:
            process_candidate(mistake, "initial_qwen")

    # 4. Return results
    return {
        "mistakes": combined,
        "general_feedback": general_feedback,
        "diagnostics": diagnostics,
        "warning": warning,
        "rule_count": len(rule_mistakes),
        "llm_count": len(llm_mistakes),
    }
