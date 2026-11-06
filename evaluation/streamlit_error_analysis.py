
"""Transparent, rule-based feedback for translation exercises."""

import re
from collections import Counter

NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|never|neither|nor|without|cannot|"
    r"can't|won't|don't|doesn't|didn't|isn't|"
    r"aren't|wasn't|weren't|couldn't|shouldn't)\b",
    re.IGNORECASE,
)

NUMBER_PATTERN = re.compile(
    r"(?<!\w)[+-]?\d+(?:[.,]\d+)*(?:%)?(?!\w)"
)

PLACEHOLDER_PATTERN = re.compile(
    r"%\d+|%\{[A-Za-z_]\w*\}|"
    r"\{[A-Za-z_]\w*\}|\$\{[A-Za-z_]\w*\}"
)

WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "be", "been", "being", "to", "of", "in",
    "on", "at", "for", "with", "and", "or",
    "that", "this", "it", "as", "by", "from",
}


def compare_tokens(reference, prediction, pattern):
    expected = Counter(pattern.findall(reference))
    actual = Counter(pattern.findall(prediction))

    return {
        "missing": list((expected - actual).elements()),
        "extra": list((actual - expected).elements()),
    }


def feedback(category, status, explanation, rule):
    return {
        "category": category,
        "status": status,
        "explanation": explanation,
        "rule": rule,
        "verified_error": False,
    }


def analyze_translation(
    source,
    reference,
    prediction,
    glossary=None,
):
    """Return suggestions, never verified error judgments."""
    results = []

    # 1. Possible omissions
    reference_words = {
        word.lower()
        for word in WORD_PATTERN.findall(reference)
        if word.lower() not in STOPWORDS and len(word) > 3
    }
    prediction_words = {
        word.lower()
        for word in WORD_PATTERN.findall(prediction)
    }

    missing = sorted(reference_words - prediction_words)

    results.append(feedback(
        "omissions",
        "review_suggested" if missing else "no_difference_detected",
        (
            "Reference terms not found verbatim: "
            + ", ".join(missing[:8])
            + ". Paraphrases may still be correct."
            if missing
            else "No missing reference content words detected."
        ),
        "Reference content-word overlap",
    ))

    # 2. Meaning and negation
    ref_negation = bool(NEGATION_PATTERN.search(reference))
    pred_negation = bool(NEGATION_PATTERN.search(prediction))

    mismatch = ref_negation != pred_negation

    results.append(feedback(
        "meaning_negation",
        "review_suggested" if mismatch else "not_verified",
        (
            "Negation appears in only one translation. "
            "Check for a possible meaning reversal."
            if mismatch
            else "No simple negation mismatch detected. "
                 "Meaning has not been verified."
        ),
        "Negation-marker presence",
    ))

    # 3. Terminology
    issues = []
    for german_term, english_term in (glossary or {}).items():
        if german_term.casefold() in source.casefold():
            if english_term.casefold() not in prediction.casefold():
                issues.append(f"{german_term} -> {english_term}")

    results.append(feedback(
        "terminology",
        "review_suggested" if issues else "not_verified",
        (
            "Possible terminology differences: " + ", ".join(issues)
            if issues
            else "No glossary conflict detected; terminology not verified."
        ),
        "Applicable glossary terms",
    ))

    # 4. Grammar
    results.append(feedback(
        "grammar",
        "not_verified",
        "Grammar is not reliably assessed by these rules. "
        "A linguistic review may be needed.",
        "No automatic grammar judgment",
    ))

    # 5. Numbers
    numbers = compare_tokens(reference, prediction, NUMBER_PATTERN)
    number_mismatch = bool(numbers["missing"] or numbers["extra"])

    results.append(feedback(
        "numbers",
        "review_suggested" if number_mismatch else "no_difference_detected",
        (
            f"Missing: {numbers['missing']}; "
            f"additional: {numbers['extra']}"
        ),
        "Numerical-token comparison",
    ))

    # 6. Placeholders
    placeholders = compare_tokens(
        source, prediction, PLACEHOLDER_PATTERN
    )
    placeholder_mismatch = bool(
        placeholders["missing"] or placeholders["extra"]
    )

    results.append(feedback(
        "placeholders",
        "review_suggested"
        if placeholder_mismatch else "no_difference_detected",
        (
            f"Missing: {placeholders['missing']}; "
            f"additional: {placeholders['extra']}"
        ),
        "Source placeholder preservation",
    ))

    # 7. Formatting
    differences = []

    if reference.count("\n") != prediction.count("\n"):
        differences.append("Line-break count differs")

    if reference.endswith(":") != prediction.endswith(":"):
        differences.append("Final colon differs")

    results.append(feedback(
        "formatting",
        "review_suggested"
        if differences else "no_difference_detected",
        (
            "; ".join(differences)
            if differences
            else "No checked formatting differences detected."
        ),
        "Line breaks and final-colon comparison",
    ))

    return results

