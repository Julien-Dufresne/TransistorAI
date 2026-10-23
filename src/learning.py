
import re
from difflib import SequenceMatcher

# Five exercises for the first MVP.
EXERCISES = [
    {
        "id": "E001",
        "english": "The application crashed after the update.",
        "focus": "Past tense and technical vocabulary",
    },
    {
        "id": "E002",
        "english": "I have been learning German for two years.",
        "focus": "Present tense and time expressions",
    },
    {
        "id": "E003",
        "english": "If I had more time, I would read more books.",
        "focus": "Conditional sentences",
    },
    {
        "id": "E004",
        "english": "The model was trained using German sentences.",
        "focus": "Passive voice",
    },
    {
        "id": "E005",
        "english": "Could you explain why the translation is incorrect?",
        "focus": "Polite requests and subordinate clauses",
    },
]


def compare_translations(user_text, model_text):
    """Compare word sequences as a preliminary diagnostic."""

    # Extract words, ignoring case and punctuation.
    user_words = re.findall(r"\w+", user_text.lower())
    model_words = re.findall(r"\w+", model_text.lower())

    # SequenceMatcher measures surface-level similarity.
    # It does not measure semantic correctness.
    similarity = SequenceMatcher(
        None, user_words, model_words
    ).ratio()

    missing = list(
        (set(model_words) - set(user_words))
    )

    extra = list(
        (set(user_words) - set(model_words))
    )

    return {
        "similarity": round(similarity * 100, 1),
        "missing": sorted(missing),
        "extra": sorted(extra),
    }
