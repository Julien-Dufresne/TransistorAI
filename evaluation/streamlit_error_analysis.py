
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


def detect_mistakes(source, reference, prediction):
    """Generate individual, unverified mistake suggestions."""
    suggestions = []
    seen = set()

    def add(category, original, corrected, explanation):
        key = (category, original, corrected)
        if key not in seen:
            seen.add(key)
            suggestions.append({
                "category": category,
                "original_text": original,
                "corrected_text": corrected,
                "explanation": explanation,
                "verified_error": False,
            })

    # Capitalization differences
    ref_words = WORD_PATTERN.findall(reference)
    pred_words = WORD_PATTERN.findall(prediction)

    if [w.lower() for w in ref_words] == [
        w.lower() for w in pred_words
    ]:
        for expected, actual in zip(ref_words, pred_words):
            if expected != actual:
                add(
                    "capitalization",
                    actual,
                    expected,
                    "Capitalization differs from the reference."
                )

    # Numerical differences
    numbers = compare_tokens(
        reference, prediction, NUMBER_PATTERN
    )

    if numbers["missing"] or numbers["extra"]:
        add(
            "numbers",
            ", ".join(numbers["extra"]),
            ", ".join(numbers["missing"]),
            "Numerical values differ from the reference."
        )

    # Placeholder preservation
    placeholders = compare_tokens(
        source, prediction, PLACEHOLDER_PATTERN
    )

    if placeholders["missing"] or placeholders["extra"]:
        add(
            "placeholders",
            ", ".join(placeholders["extra"]),
            ", ".join(placeholders["missing"]),
            "Placeholders should be preserved from the source."
        )

    # Negation differences
    ref_negation = bool(NEGATION_PATTERN.search(reference))
    pred_negation = bool(NEGATION_PATTERN.search(prediction))

    if ref_negation != pred_negation:
        add(
            "meaning",
            prediction,
            reference,
            "Possible change in meaning due to negation."
        )

    # Possible omissions
    reference_words = {
        w.lower()
        for w in WORD_PATTERN.findall(reference)
        if w.lower() not in STOPWORDS and len(w) > 3
    }

    prediction_words = {
        w.lower() for w in WORD_PATTERN.findall(prediction)
    }

    missing = sorted(reference_words - prediction_words)

    for word in missing:
        add(
            "omissions",
            "",
            word,
            "Reference word not found. Check whether "
            "the meaning is already expressed differently."
        )

    # Formatting
    if reference.count("\n") != prediction.count("\n"):
        add(
            "formatting",
            prediction,
            reference,
            "Line-break count differs from the reference."
        )

    return suggestions



ENGLISH_NUMBERS = {
    "zero": 0, "one": 1, "two": 2, "three": 3,
    "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10,
}

GERMAN_NUMBERS = {
    "null": 0, "eins": 1, "ein": 1, "eine": 1,
    "zwei": 2, "drei": 3, "vier": 4,
    "fünf": 5, "sechs": 6, "sieben": 7,
    "acht": 8, "neun": 9, "zehn": 10,
}


def detect_structured_mistakes(source, reference, prediction):
    """Detect high-confidence capitalization and number errors."""

    mistakes = []

    # 1. Sentence-initial capitalization
    first_word = re.search(r"[A-Za-z]+", prediction)

    if first_word:
        original = first_word.group()

        if original[0].islower():
            corrected = original[0].upper() + original[1:]

            mistakes.append({
                "category": "capitalization",
                "original_text": original,
                "corrected_text": corrected,
                "explanation": (
                    "The first word of an English sentence "
                    "must be capitalized."
                ),
                "verified_error": False,
                "detector": "rules",
            })

    # 2. Compare written and digit numbers
    def extract_numbers(text, language):
        number_map = (
            GERMAN_NUMBERS
            if language == "de"
            else ENGLISH_NUMBERS
        )

        tokens = re.findall(
            r"\b[\wäöüÄÖÜß]+\b|\d+",
            text.lower(),
        )

        values = []

        for token in tokens:
            if token.isdigit():
                values.append((token, int(token)))
            elif token in number_map:
                values.append((token, number_map[token]))

        return values

    source_numbers = extract_numbers(source, "de")
    prediction_numbers = extract_numbers(prediction, "en")

    # Only compare simple one-to-one number cases.
    if len(source_numbers) == len(prediction_numbers) == 1:

        source_word, source_value = source_numbers[0]
        prediction_word, prediction_value = prediction_numbers[0]

        if source_value != prediction_value:

            reference_numbers = extract_numbers(reference, "en")

            corrected = (
                reference_numbers[0][0]
                if len(reference_numbers) == 1
                and reference_numbers[0][1] == source_value
                else str(source_value)
            )

            mistakes.append({
                "category": "numbers",
                "original_text": prediction_word,
                "corrected_text": corrected,
                "explanation": (
                    f"The German source specifies {source_value}, "
                    f"but your translation says {prediction_value}."
                ),
                "verified_error": False,
                "detector": "rules",
            })

    return mistakes


def detect_duration_preposition(prediction):
    """Detect 'since' followed by a simple duration."""

    duration_pattern = re.compile(
        r"\bsince\s+"
        r"(?:(?:\d+|one|two|three|four|five|six|"
        r"seven|eight|nine|ten|a|an)\s+)"
        r"(?:days?|weeks?|months?|years?|hours?|minutes?)\b",
        re.IGNORECASE,
    )

    suggestions = []

    for match in duration_pattern.finditer(prediction):
        original = match.group()
        corrected = re.sub(
            r"^since\b", "for", original,
            flags=re.IGNORECASE,
        )

        suggestions.append({
            "category": "prepositions",
            "original_text": original,
            "corrected_text": corrected,
            "explanation": (
                "Use 'for' with a duration, such as "
                "'for five years'. Use 'since' with a "
                "starting point, such as 'since 2021'."
            ),
            "verified_error": False,
            "detector": "rules",
        })

    return suggestions


def detect_missing_negation(source, reference, prediction):
    """Detect a missing English negation when supported by German."""

    german_negation = re.compile(
        r"\b(?:nicht|kein|keine|keinen|keinem|keiner|keines|nie|niemals)\b",
        re.IGNORECASE,
    )

    english_negation = re.compile(
        r"\b(?:not|no|never|don't|doesn't|didn't|"
        r"isn't|aren't|wasn't|weren't|can't|cannot|"
        r"won't|shouldn't|couldn't|without)\b",
        re.IGNORECASE,
    )

    if not german_negation.search(source):
        return []

    if not english_negation.search(reference):
        return []

    if english_negation.search(prediction):
        return []

    return [{
        "category": "meaning",
        "original_text": prediction,
        "corrected_text": reference,
        "explanation": (
            "The German source contains a negation, "
            "but the English translation does not. "
            "This may reverse the intended meaning."
        ),
        "verified_error": False,
        "detector": "rules",
    }]
