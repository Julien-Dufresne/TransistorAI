
"""
Day 19 — German vocabulary quiz utilities.

Normalizes English answers and checks them against
the vocabulary bank.
"""

import re
import unicodedata



def normalize_answer(answer):
    """
    Normalize English vocabulary answers.

    Ignore capitalization, punctuation and an optional
    leading English article (a, an, the).
    """
    answer = unicodedata.normalize("NFKC", answer)
    answer = answer.casefold().strip()
    answer = re.sub(r"[^\w\s]", " ", answer)
    answer = " ".join(answer.split())

    # Allow optional English articles at the beginning.
    answer = re.sub(r"^(the|a|an)\s+", "", answer)

    return answer



def check_answer(user_answer, expected_answer):
    """
    Compare a learner's answer with accepted translations.

    Alternatives are separated by ' or ' in the vocabulary bank.
    """
    normalized_user = normalize_answer(user_answer)

    if not normalized_user:
        return False

    alternatives = re.split(
        r"\s+or\s+",
        expected_answer,
        flags=re.IGNORECASE,
    )

    return any(
        normalized_user == normalize_answer(alternative)
        for alternative in alternatives
    )
