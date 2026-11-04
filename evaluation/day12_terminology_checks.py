
# Import regular expressions for controlled term matching.
import re


def contains_term(text: str, term: str) -> bool:
    """
    Check whether a term occurs as a complete word or phrase.

    This prevents short terms from accidentally matching
    parts of longer words.
    """

    # Escape special characters in the terminology entry.
    escaped_term = re.escape(term)

    # Require word boundaries around the term or phrase.
    pattern = rf"(?<!\w){escaped_term}(?!\w)"

    # Match without distinguishing uppercase and lowercase.
    return re.search(pattern, text, re.IGNORECASE) is not None


def check_terminology(
    source: str,
    prediction: str,
    glossary: list[dict]
) -> list[dict]:
    """
    Identify glossary terms present in the source whose
    approved translations are missing from the prediction.
    """

    # Store every applicable terminology check.
    results = []

    # Evaluate each approved terminology entry.
    for entry in glossary:

        # Ignore terms that do not occur in the German source.
        if not contains_term(source, entry["source_de"]):
            continue

        # Include the preferred English translation.
        approved = [entry["target_en"]]

        # Include alternative approved translations, if any.
        alternatives = entry.get("alternatives_en", "")

        if alternatives:
            approved.extend(alternatives.split("|"))

        # Remove whitespace from all approved translations.
        approved = [term.strip() for term in approved if term.strip()]

        # A term passes if at least one approved translation appears.
        passed = any(
            contains_term(prediction, term)
            for term in approved
        )

        # Save the outcome for later aggregation and review.
        results.append({
            "term_id": entry["term_id"],
            "source_term": entry["source_de"],
            "approved_targets": approved,
            "passed": passed,
        })

    return results


# Counter allows us to compare tokens including duplicates.
from collections import Counter

# Import regular expressions to identify placeholder syntax.
import re



def extract_placeholders(text: str) -> list[str]:
    """
    Extract supported placeholders while avoiding ordinary
    percentages and translatable LaTeX arguments.

    Supported formats:
    - Named placeholders: {device_id}
    - Named printf: %(name)s
    - Positional printf: %s, %d, %f
    - Qt placeholders: %1, %2, %L1

    Known limitations:
    - LaTeX command arguments are not treated as placeholders.
    - Some advanced formatting syntaxes are not yet supported.
    """

    # Match named identifiers inside braces.
    # Restrict to ASCII identifiers and require at least one
    # underscore or digit to reduce false positives.
    named_braces = (
        r"\{(?=[A-Za-z0-9_]*[_0-9])"
        r"[A-Za-z_][A-Za-z0-9_]*\}"
    )

    # Match named printf-style placeholders.
    named_printf = (
        r"%\([A-Za-z_][A-Za-z0-9_]*\)[sdf]"
    )

    # Match simple printf placeholders.
    # Do not allow whitespace after the percent sign.
    simple_printf = r"%(?:\d+\$)?[sdf]"

    # Match Qt placeholders such as %1 and %L1.
    qt_placeholder = r"%L?[1-9]\d*"

    # Combine all supported formats.
    pattern = (
        rf"{named_braces}"
        rf"|{named_printf}"
        rf"|{simple_printf}"
        rf"|{qt_placeholder}"
    )

    # Return all matching placeholders.
    return re.findall(pattern, text)



def check_placeholders(source: str, prediction: str) -> dict:
    """
    Verify that the prediction preserves the same placeholders,
    including the number of occurrences of each placeholder.
    """

    # Extract placeholders from both texts.
    source_tokens = extract_placeholders(source)
    predicted_tokens = extract_placeholders(prediction)

    # Compare token frequencies, ignoring their order.
    passed = Counter(source_tokens) == Counter(predicted_tokens)

    # Return a structured result for reporting.
    return {
        "passed": passed,
        "source_tokens": source_tokens,
        "predicted_tokens": predicted_tokens,
    }


# Import Counter to compare repeated numeric values.
from collections import Counter

# Import Decimal to compare numeric values precisely.
from decimal import Decimal, InvalidOperation

# Import regular expressions for numeric extraction.
import re


def _canonical_number(value: str) -> str:
    """
    Normalize a numeric value into a comparable representation.

    Handles decimal commas, decimal points and common grouping
    conventions. Ambiguous single separators are interpreted
    conservatively.
    """

    value = value.strip()

    # Handle numbers containing both separator types.
    if "," in value and "." in value:
        if value.rfind(",") > value.rfind("."):
            value = value.replace(".", "").replace(",", ".")
        else:
            value = value.replace(",", "")

    # Interpret one comma followed by three digits as grouping.
    elif re.fullmatch(r"\d{1,3}(?:,\d{3})+", value):
        value = value.replace(",", "")

    # Interpret one period followed by three digits as grouping.
    elif re.fullmatch(r"\d{1,3}(?:\.\d{3})+", value):
        value = value.replace(".", "")

    # Interpret remaining commas as decimal separators.
    else:
        value = value.replace(",", ".")

    try:
        return str(Decimal(value).normalize())
    except InvalidOperation:
        return value


def _extract_numeric_tokens(text: str) -> list[str]:
    """
    Extract comparable numeric tokens from a sentence.

    Special cases:
    - German and English ordinals
    - 24-hour and 12-hour clock times
    - Currency amounts with million/m abbreviations

    Placeholder tokens are excluded from ordinary numeric checks.
    """

    tokens = []

    # Remove protected placeholders before processing.
    cleaned = text

    for placeholder in extract_placeholders(text):
        cleaned = cleaned.replace(placeholder, " ")

    # Normalize equivalent time expressions.
    # Example: 15.00 Uhr -> TIME:15:00
    german_time_pattern = (
        r"\b([01]?\d|2[0-3])[.:]([0-5]\d)\s*Uhr\b"
    )

    def replace_german_time(match):
        hour = int(match.group(1))
        minute = int(match.group(2))
        tokens.append(f"TIME:{hour:02d}:{minute:02d}")
        return " "

    cleaned = re.sub(
        german_time_pattern,
        replace_german_time,
        cleaned,
        flags=re.IGNORECASE,
    )

    # Normalize English a.m./p.m. times.
    # Example: 3 p.m. -> TIME:15:00
    english_time_pattern = (
        r"\b(1[0-2]|0?[1-9])"
        r"(?::([0-5]\d))?"
        r"\s*([ap])\.?\s*m\.?"
    )

    def replace_english_time(match):
        hour = int(match.group(1))
        minute = int(match.group(2) or 0)
        period = match.group(3).lower()

        # Convert 12-hour clock time into 24-hour format.
        hour = hour % 12

        if period == "p":
            hour += 12

        tokens.append(f"TIME:{hour:02d}:{minute:02d}")
        return " "

    cleaned = re.sub(
        english_time_pattern,
        replace_english_time,
        cleaned,
        flags=re.IGNORECASE,
    )

    # Normalize currency amounts expressed in millions.
    # Example: £11 Millionen -> MONEY:GBP:11000000
    currency_pattern = (
        r"£\s*(\d+(?:[.,]\d+)?)"
        r"\s*(Millionen|million|millions|m)\b"
    )

    def replace_currency(match):
        amount = Decimal(
            _canonical_number(match.group(1))
        )

        total = amount * Decimal("1000000")

        tokens.append(
            f"MONEY:GBP:{str(total.normalize())}"
        )

        return " "

    cleaned = re.sub(
        currency_pattern,
        replace_currency,
        cleaned,
        flags=re.IGNORECASE,
    )

    # Normalize German and English ordinal expressions.
    ordinal_pattern = (
        r"(?<!\w)(\d+)(?:st|nd|rd|th)\b"
        r"|(?<!\w)(\d+)\.(?=\s|$)"
    )

    def replace_ordinal(match):
        value = match.group(1) or match.group(2)

        tokens.append(
            f"NUMBER:{_canonical_number(value)}"
        )

        return " "

    cleaned = re.sub(
        ordinal_pattern,
        replace_ordinal,
        cleaned,
        flags=re.IGNORECASE,
    )

    # Exclude software version strings from quantity checks.
    # These will require their own exact-preservation check.
    cleaned = re.sub(
        r"(?<![\w.])\d+(?:\.\d+){2,}(?![\w.])",
        " ",
        cleaned,
    )

    # Extract remaining numeric expressions.
    number_pattern = (
        r"(?<![\w%])"
        r"[-+]?\d+(?:[.,]\d+)*"
        r"(?![\w])"
    )

    for match in re.finditer(number_pattern, cleaned):
        value = match.group(0)

        tokens.append(
            f"NUMBER:{_canonical_number(value)}"
        )

    return tokens


def extract_numbers(text: str) -> list[str]:
    """
    Return comparable numeric tokens from a sentence.

    Kept as a public function for compatibility with existing
    tests and evaluation code.
    """

    return _extract_numeric_tokens(text)


def normalize_number(value: str) -> str:
    """
    Normalize ordinary numeric values.

    This public wrapper preserves the existing API.
    """

    return _canonical_number(value)


def check_numbers(source: str, prediction: str) -> dict:
    """
    Compare numeric content between source and prediction.

    Repeated values are counted, so missing or duplicated
    numbers can be detected.
    """

    # Extract normalized numeric tokens from both texts.
    source_numbers = extract_numbers(source)
    prediction_numbers = extract_numbers(prediction)

    # Compare the number of occurrences of each token.
    passed = (
        Counter(source_numbers)
        == Counter(prediction_numbers)
    )

    # Return the same fields expected by the evaluation pipeline.
    return {
        "passed": passed,
        "source_numbers": source_numbers,
        "prediction_numbers": prediction_numbers,
        "source_normalized": source_numbers,
        "prediction_normalized": prediction_numbers,
    }




# Import numeric extraction for additional regression tests.
from evaluation.day12_terminology_checks import extract_numbers


def test_qt_placeholder_excluded_from_numbers():
    """Qt placeholders must not count as numeric values."""

    # A Qt placeholder is handled by placeholder validation.
    result = extract_numbers("Lautstärke auf %1 stellen")

    # No ordinary numeric values should be detected.
    assert result == []


def test_version_excluded_from_numbers():
    """Version identifiers must not be treated as quantities."""

    # A software version is not an ordinary numeric quantity.
    result = extract_numbers("Version 3.12.4 installieren")

    # The version is excluded from numeric comparison.
    assert result == []


def test_numeric_value_with_placeholder():
    """Real numbers must still be detected near placeholders."""

    # The placeholder is ignored, but 24 is retained.
    result = extract_numbers("Gerät %1 benötigt 24 V")

    # Verify that only the numeric quantity remains.
    assert result == ["24"]


import re


def normalize_ordinal(value: str) -> str:
    """
    Normalize German and English ordinal expressions.

    Examples:
    - 89.   -> 89
    - 89th  -> 89
    - 125th -> 125
    - 27.   -> 27

    This function handles ordinal tokens only.
    """

    # Remove surrounding whitespace.
    value = value.strip()

    # Match English ordinal suffixes.
    english_ordinal = re.fullmatch(
        r"(\d+)(st|nd|rd|th)",
        value,
        flags=re.IGNORECASE,
    )

    # Return the numeric component if an ordinal is found.
    if english_ordinal:
        return english_ordinal.group(1)

    # Match German ordinals ending with a period.
    german_ordinal = re.fullmatch(
        r"(\d+)\.",
        value,
    )

    # Return the numeric component if matched.
    if german_ordinal:
        return german_ordinal.group(1)

    # Return the original value if no ordinal pattern matches.
    return value


from collections import Counter


# Units whose symbols should normally remain unchanged.
# Case matters: MB and mb are not equivalent.
TECHNICAL_UNITS = (
    "kHz", "MHz", "GHz",
    "kW", "mW",
    "kV", "mV",
    "GB", "MB", "KB", "TB",
    "Hz", "ms", "kg",
    "V", "A", "W",
)


def extract_units(text: str) -> list[str]:
    """
    Extract supported technical units attached to numbers.

    Examples:
    - 24 V
    - 3.5 GHz
    - 512 MB

    This deliberately excludes ordinary words and units
    without a preceding numeric value.
    """

    # Match longer unit symbols before shorter ones.
    units_pattern = "|".join(
        re.escape(unit)
        for unit in sorted(
            TECHNICAL_UNITS,
            key=len,
            reverse=True,
        )
    )

    # Require a number immediately before the unit,
    # allowing optional whitespace.
    pattern = (
        r"(?<![\w])"
        r"[-+]?\d+(?:[.,]\d+)*"
        r"\s*"
        rf"({units_pattern})"
        r"(?!\w)"
    )

    # Return the unit symbols without numeric values.
    return re.findall(pattern, text)


def check_units(source: str, prediction: str) -> dict:
    """
    Compare technical unit occurrences in source and prediction.

    The check detects missing, changed or duplicated units.
    It does not yet validate numerical unit conversions.
    """

    # Extract units from both sentences.
    source_units = extract_units(source)
    prediction_units = extract_units(prediction)

    # Compare unit frequencies, ignoring their order.
    passed = (
        Counter(source_units)
        == Counter(prediction_units)
    )

    # Return a structured evaluation result.
    return {
        "passed": passed,
        "source_units": source_units,
        "prediction_units": prediction_units,
    }


# ============================================================
# DAY 12 — PROTECTED URL AND FILE-PATH CHECKS
# ============================================================

import re
from collections import Counter


# Detect HTTP and HTTPS URLs.
URL_PATTERN = re.compile(
    r"https?://[^\s<>\"']+"
)

# Detect absolute, home-relative and current-directory paths.
# Require a meaningful path component after the prefix.


# ============================================================
# DAY 12 — REFINED FILE-PATH DETECTION
# ============================================================

# Detect paths with an explicit filesystem prefix.
#
# Supported examples:
#   /usr/bin/perl
#   ~/.kde/share/apps/kmplayer/intro.xml
#   ./configure
#   /lib/kde3/plugins
#
# The negative lookbehind prevents matching paths that
# begin inside ordinary words, including "Ein-/Ausgabe".

PATH_PATTERN = re.compile(
    r"(?<![\w./~-])"
    r"(?:~/|\./|/)"
    r"(?:[\w.-]+/)*"
    r"[\w.-]+"
)


def extract_file_paths(text: str) -> list[str]:
    """
    Extract filesystem paths from technical text.

    This detector deliberately prioritizes precision.

    It excludes:
        - URL components
        - German compounds containing slashes
        - slash-only punctuation
        - path-like strings inside ordinary words

    It preserves:
        - absolute paths
        - home-relative paths
        - current-directory paths
    """

    # Remove URLs before searching for filesystem paths.
    text_without_urls = URL_PATTERN.sub(" ", text)

    # Find candidate filesystem paths.
    matches = PATH_PATTERN.findall(text_without_urls)

    paths = []

    for match in matches:

        # Remove punctuation that may terminate a sentence.
        cleaned = match.rstrip(".,;:!?")

        # Reject meaningless paths.
        if cleaned in {"/", "/.", "./", "~/", "/.."}:
            continue

        # Reject standalone slash-prefixed words.
        # These are ambiguous and frequently represent
        # linguistic constructions rather than file paths.
        #
        # Keep explicit ./ and ~/ paths.
        if (
            cleaned.startswith("/")
            and cleaned.count("/") == 1
        ):
            continue

        # Preserve the extracted path.
        paths.append(cleaned)

    return paths


def extract_urls(text: str) -> list[str]:
    """
    Extract URLs while excluding trailing sentence punctuation.
    """

    # Find candidate URLs in the sentence.
    matches = URL_PATTERN.findall(text)

    # Remove punctuation that commonly ends a sentence.
    return [
        match.rstrip(".,;:!?")
        for match in matches
    ]




def check_protected_content(
    source: str,
    prediction: str,
    category: str,
) -> dict:
    """
    Compare protected strings between source and prediction.

    Supported categories:
        - urls
        - file_paths

    Each occurrence must be preserved exactly.
    """

    # Select the appropriate extraction function.
    extractors = {
        "urls": extract_urls,
        "file_paths": extract_file_paths,
    }

    # Reject unsupported categories.
    if category not in extractors:
        raise ValueError(
            f"Unsupported protected category: {category}"
        )

    # Extract strings from both sentences.
    extractor = extractors[category]

    source_tokens = extractor(source)
    prediction_tokens = extractor(prediction)

    # Compare occurrence counts, including duplicates.
    passed = (
        Counter(source_tokens)
        == Counter(prediction_tokens)
    )

    # Return the result in the standard evaluation format.
    return {
        "passed": passed,
        "source_tokens": source_tokens,
        "prediction_tokens": prediction_tokens,
    }
