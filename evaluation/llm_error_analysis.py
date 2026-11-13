
"""Local Qwen-based translation error analysis."""

import json
import re


ALLOWED_CATEGORIES = {
    "grammar", "conjugation", "plurals",
    "capitalization", "omissions", "word_order",
    "articles", "prepositions", "vocabulary",
    "spelling", "meaning", "terminology",
    "numbers", "formatting", "other",
}



def build_analysis_prompt(source, reference, prediction):
    import json

    data = {
        "source": source,
        "reference": reference,
        "student_translation": prediction,
    }

    instructions = """
You are an expert German-to-English translation examiner.

Your task is to identify genuine mistakes in a student's
English translation.

The German source is authoritative.
The English reference is only ONE possible correct translation.

ANALYSIS PROCEDURE:

1. MEANING
Check whether the translation preserves the German meaning.
Pay special attention to:
- Negation: missing or incorrectly added negatives
- Conjunctions: although, because, if, unless, despite
- Time references: past, present, future
- Missing or added information
- Incorrect vocabulary

2. GRAMMAR
Check independently:
- Verb tense and aspect
- Conjugation
- Word order
- Articles and prepositions
- Singular and plural forms

3. VALID TRANSLATIONS
Accept natural paraphrases.
Do not flag differences that preserve the source meaning.
Do not treat the reference as the only valid answer.

4. CORRECTIONS
Return a separate item for each independent mistake.
Each correction must:
- Identify a meaningful phrase from the student's text
- Provide a grammatically complete replacement phrase
- Preserve the meaning of the German source
- Avoid changing unrelated words
- Avoid overlapping another correction where possible

5. EXPLANATIONS
Explain the actual mistake, not merely the difference
from the reference.
Never invent grammatical rules.
If uncertain, omit the suggestion.

6. CATEGORIES
Use only:
grammar, conjugation, plurals, omissions, word_order,
articles, prepositions, vocabulary, spelling, meaning,
terminology, other.

Capitalization and numbers are checked separately.
Do not report them.

7. OUTPUT
Return ONLY valid JSON in this structure:

{
  "mistakes": [
    {
      "category": "meaning",
      "original_text": "incorrect phrase",
      "corrected_text": "corrected phrase",
      "explanation": "Why this is incorrect"
    }
  ]
}

If the translation is acceptable, return:
{"mistakes": []}
"""

    return (
        instructions
        + "\n\nINPUT:\n"
        + json.dumps(data, ensure_ascii=False)
    )




def parse_analysis_response(response, prediction):
    """Validate Qwen's JSON before displaying suggestions."""

    cleaned = response.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        raise ValueError("Qwen returned invalid JSON.")

    if not isinstance(parsed, dict):
        raise ValueError("Expected a JSON object.")

    items = parsed.get("mistakes")

    if not isinstance(items, list):
        raise ValueError("Expected a mistakes list.")

    suggestions = []
    seen = set()

    for item in items[:20]:
        if not isinstance(item, dict):
            continue

        category = item.get("category", "other")
        original = item.get("original_text", "")
        corrected = item.get("corrected_text", "")
        explanation = item.get("explanation", "")

        if category not in ALLOWED_CATEGORIES:
            category = "other"

        if not all(
            isinstance(value, str)
            for value in (original, corrected, explanation)
        ):
            continue

        original = original.strip()
        corrected = corrected.strip()
        explanation = explanation.strip()

        if not original or not corrected or not explanation:
            continue

        # The proposed error must actually appear
        # in the student's translation.
        if original.casefold() not in prediction.casefold():
            continue

        # Exclude categories handled by deterministic rules.
        if category in {"capitalization", "numbers"}:
            continue

        if original.casefold() == corrected.casefold():
            if category != "capitalization":
                continue

        key = (category, original.casefold(), corrected.casefold())

        if key in seen:
            continue

        seen.add(key)

        suggestions.append({
            "category": category,
            "original_text": original,
            "corrected_text": corrected,
            "explanation": explanation,
            "verified_error": False,
            "detector": "qwen",
        })

    return suggestions


def analyze_with_qwen(source, reference, prediction):
    """Run local Qwen and return validated suggestions."""

    from mlx_lm import load, generate

    model_id = "mlx-community/Qwen2.5-3B-Instruct-4bit"

    model, tokenizer = load(model_id)

    system_message = (
        "You are a precise linguistic error analyst. "
        "Return valid JSON only."
    )

    messages = [
        {"role": "system", "content": system_message},
        {
            "role": "user",
            "content": build_analysis_prompt(
                source, reference, prediction
            ),
        },
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    response = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=768,
        verbose=False,
    )

    return parse_analysis_response(response, prediction)
