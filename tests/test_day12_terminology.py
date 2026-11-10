
# Import the functions that we want to test.
from evaluation.day12_terminology_checks import (
    check_terminology,
    check_placeholders,
)


def test_correct_terminology():
    """An approved translation should pass."""

    # Define a small test glossary.
    glossary = [{
        "term_id": "T010",
        "source_de": "Datenbank",
        "target_en": "database",
        "alternatives_en": "",
    }]

    # Evaluate a correct technical translation.
    results = check_terminology(
        "Öffnen Sie die Datenbank.",
        "Open the database.",
        glossary,
    )

    # Verify that the terminology check passes.
    assert len(results) == 1
    assert results[0]["passed"] is True


def test_incorrect_terminology():
    """A missing approved translation should be flagged."""

    # Define the same approved terminology entry.
    glossary = [{
        "term_id": "T010",
        "source_de": "Datenbank",
        "target_en": "database",
        "alternatives_en": "",
    }]

    # Evaluate a prediction without the approved term.
    results = check_terminology(
        "Öffnen Sie die Datenbank.",
        "Open the storage.",
        glossary,
    )

    # Verify that the checker flags the prediction.
    assert results[0]["passed"] is False


def test_preserved_placeholder():
    """An unchanged placeholder should pass."""

    # The placeholder appears in both source and prediction.
    result = check_placeholders(
        "Gerät {device_id} ist bereit.",
        "Device {device_id} is ready.",
    )

    # Verify successful preservation.
    assert result["passed"] is True


def test_missing_placeholder():
    """A removed placeholder should fail."""

    # The English prediction omits the device identifier.
    result = check_placeholders(
        "Gerät {device_id} ist bereit.",
        "The device is ready.",
    )

    # Verify that the missing placeholder is detected.
    assert result["passed"] is False


def test_duplicate_placeholder():
    """Duplicated placeholders should also be flagged."""

    # The prediction contains the placeholder twice.
    result = check_placeholders(
        "Gerät {device_id} ist bereit.",
        "Device {device_id} ({device_id}) is ready.",
    )

    # Verify that duplicate occurrences are detected.
    assert result["passed"] is False


def test_percentage_not_placeholder():
    """Ordinary percentage expressions must not be placeholders."""

    # Import the extraction function.
    from evaluation.day12_terminology_checks import (
        extract_placeholders,
    )

    # Verify that German percentage text is ignored.
    assert extract_placeholders("70–90 % der Männer") == []

    # Verify that a standalone percentage is ignored.
    assert extract_placeholders("% der Originalgröße") == []


def test_qt_placeholder():
    """Qt positional placeholders must be preserved."""

    # Evaluate a correct Qt translation.
    result = check_placeholders(
        "Lautstärke auf %1% stellen",
        "Set volume to %1%",
    )

    # Verify successful preservation.
    assert result["passed"] is True


def test_qt_placeholder_missing():
    """Removing a Qt placeholder must be detected."""

    # Evaluate a prediction that omits the placeholder.
    result = check_placeholders(
        "Lautstärke auf %1% stellen",
        "Set the volume",
    )

    # Verify that the missing placeholder is detected.
    assert result["passed"] is False


def test_latex_title_not_placeholder():
    """A translatable LaTeX argument is not a protected token."""

    # Import the extraction function.
    from evaluation.day12_terminology_checks import (
        extract_placeholders,
    )

    # Verify that the title argument is not extracted.
    assert extract_placeholders(
        r"\subsubsection{Titel}"
    ) == []


# Import the new numeric checking functions.
from evaluation.day12_terminology_checks import (
    check_numbers,
    normalize_number,
)


def test_numbers_preserved():
    """Identical numeric values should pass."""

    # Evaluate a correct technical translation.
    result = check_numbers(
        "Die Spannung beträgt 24 V.",
        "The voltage is 24 V.",
    )

    # Verify that the numeric value is preserved.
    assert result["passed"] is True


def test_number_changed():
    """Changed numeric values should fail."""

    # The prediction changes 24 into 12.
    result = check_numbers(
        "Die Spannung beträgt 24 V.",
        "The voltage is 12 V.",
    )

    # Verify that the change is detected.
    assert result["passed"] is False


def test_decimal_separator():
    """German and English decimal separators should match."""

    # The German comma becomes an English decimal point.
    result = check_numbers(
        "Die Spannung beträgt 3,5 V.",
        "The voltage is 3.5 V.",
    )

    # Verify equivalent numeric values.
    assert result["passed"] is True


def test_duplicate_number():
    """An additional numeric occurrence should fail."""

    # The prediction repeats the numeric value.
    result = check_numbers(
        "Die Spannung beträgt 24 V.",
        "The voltage is 24 V, not 24 V.",
    )

    # Verify that duplicates are detected.
    assert result["passed"] is False


def test_normalize_grouped_number():
    """Grouped numbers should normalize consistently."""

    # Compare German and English thousands/decimal formats.
    assert normalize_number("1.000,5") == "1000.5"
    assert normalize_number("1,000.5") == "1000.5"


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
    """
    Verify that placeholders are excluded while
    ordinary numeric values are preserved.
    """

    # Extract numbers from a sentence containing
    # both a Qt placeholder and a numeric quantity.
    result = extract_numbers("Gerät %1 benötigt 24 V")

    # %1 must be excluded because it is a placeholder.
    # 24 must remain as a normalized numeric token.
    assert result == ["NUMBER:24"]



# Import the ordinal normalization function.
from evaluation.day12_terminology_checks import normalize_ordinal


def test_german_ordinal():
    """German ordinal numbers should normalize correctly."""

    # Convert the German ordinal to its numeric form.
    assert normalize_ordinal("89.") == "89"


def test_english_ordinal():
    """English ordinal numbers should normalize correctly."""

    # Convert the English ordinal to its numeric form.
    assert normalize_ordinal("89th") == "89"


def test_english_ordinal_125():
    """Three-digit English ordinals should be supported."""

    # Normalize a three-digit ordinal.
    assert normalize_ordinal("125th") == "125"


# These tests represent actual examples found during Day 12 evaluation.
# They protect against both false positives and missed numeric errors.


def test_ordinal_translation():
    """German and English ordinals should be equivalent."""

    result = check_numbers(
        "in der 89. Minute",
        "in the 89th minute",
    )

    assert result["passed"] is True


def test_street_ordinal():
    """Street numbers expressed as ordinals should match."""

    result = check_numbers(
        "in der 125. Straße",
        "on 125th Street",
    )

    assert result["passed"] is True


def test_time_conversion():
    """Equivalent 24-hour and 12-hour times should match."""

    result = check_numbers(
        "um 15.00 Uhr",
        "at 3 p.m.",
    )

    assert result["passed"] is True


def test_currency_abbreviation():
    """Million and m should represent equivalent currency amounts."""

    result = check_numbers(
        "£ 11 Millionen",
        "£11m",
    )

    assert result["passed"] is True


def test_changed_year():
    """An incorrect year must be detected."""

    result = check_numbers(
        "Government of Wales Act 2017",
        "Government of Wales Act 2004",
    )

    assert result["passed"] is False


def test_changed_identifier():
    """An altered numeric identifier must be detected."""

    result = check_numbers(
        "512, 513, 514",
        "512, 512, 514",
    )

    assert result["passed"] is False


# Import the unit checking functions.
from evaluation.day12_terminology_checks import (
    extract_units,
    check_units,
)


def test_unit_preserved():
    """An unchanged technical unit should pass."""

    result = check_units(
        "Die Spannung beträgt 24 V.",
        "The voltage is 24 V.",
    )

    assert result["passed"] is True


def test_unit_changed():
    """A changed unit should fail."""

    result = check_units(
        "Die Spannung beträgt 24 V.",
        "The voltage is 24 A.",
    )

    assert result["passed"] is False


def test_unit_missing():
    """A missing unit should fail."""

    result = check_units(
        "Die Frequenz beträgt 3.5 GHz.",
        "The frequency is 3.5.",
    )

    assert result["passed"] is False


def test_unit_duplicate():
    """Duplicated units should fail."""

    result = check_units(
        "Der Speicher beträgt 512 MB.",
        "The memory is 512 MB plus 512 MB.",
    )

    assert result["passed"] is False


def test_unit_extraction():
    """Multiple technical units should be extracted."""

    result = extract_units(
        "24 V, 3.5 GHz und 512 MB"
    )

    assert result == ["V", "GHz", "MB"]



# Import protected-content checking functions.
from evaluation.day12_terminology_checks import (
    extract_urls,
    extract_file_paths,
    check_protected_content,
)


def test_url_preserved():
    """An unchanged URL should pass."""

    result = check_protected_content(
        "Besuchen Sie http://example.org.",
        "Visit http://example.org.",
        "urls",
    )

    assert result["passed"] is True


def test_url_changed():
    """A modified URL should fail."""

    result = check_protected_content(
        "Besuchen Sie http://example.org.",
        "Visit http://example.com.",
        "urls",
    )

    assert result["passed"] is False


def test_url_trailing_punctuation():
    """Sentence-ending punctuation is not part of the URL."""

    assert extract_urls(
        "Besuchen Sie http://example.org."
    ) == ["http://example.org"]


def test_file_path_preserved():
    """An unchanged file path should pass."""

    result = check_protected_content(
        "Öffnen Sie /usr/bin/perl.",
        "Open /usr/bin/perl.",
        "file_paths",
    )

    assert result["passed"] is True


def test_file_path_modified():
    """A path containing an inserted space should fail."""

    result = check_protected_content(
        "Verzeichnis: /lib/kde3/plugins",
        "Directory: /lib/ kde3/plugins",
        "file_paths",
    )

    assert result["passed"] is False


def test_home_directory_path():
    """Preserve the leading home-directory marker."""

    assert extract_file_paths(
        "~/.kde/share/apps/kmplayer/intro.xml"
    ) == ["~/.kde/share/apps/kmplayer/intro.xml"]


def test_url_not_counted_as_file_path():
    """URL path components must not become file-path checks."""

    assert extract_file_paths(
        "http://example.org/docs/index.html"
    ) == []



# ============================================================
# DAY 12 — FILE-PATH REGRESSION TESTS
# ============================================================


def test_german_compound_not_path():
    """German compounds must not be classified as paths."""

    text = "Ein-/Ausgabemodul verwenden"

    assert extract_file_paths(text) == []


def test_german_compound_plural_not_path():
    """Plural German compounds must also be excluded."""

    text = "Weitere Ein-/Ausgabemodule installieren"

    assert extract_file_paths(text) == []


def test_absolute_multicomponent_path():
    """A genuine absolute filesystem path must be detected."""

    text = "Installieren Sie /usr/bin/perl."

    assert extract_file_paths(text) == ["/usr/bin/perl"]


def test_home_relative_path():
    """Home-directory paths must retain their prefix."""

    text = "~/.kde/share/apps/kmplayer/intro.xml"

    assert extract_file_paths(text) == [
        "~/.kde/share/apps/kmplayer/intro.xml"
    ]


def test_broken_path_fails_preservation():
    """Whitespace corruption must cause a failed check."""

    result = check_protected_content(
        "Verzeichnis: /lib/kde3/plugins",
        "Directory: /lib/ kde3/plugins",
        "file_paths",
    )

    assert result["passed"] is False

