
"""Extract individual translation corrections from broad AI feedback."""

import json

from evaluation.llm_error_analysis import parse_analysis_response


def build_extraction_prompt(source, reference, prediction, feedback):
    data = {
        "german_source": source,
        "reference_translation": reference,
        "student_translation": prediction,
        "general_feedback": feedback.get("explanation", ""),
    }

    instructions = """
You are a German-to-English translation error analyst.

The German source determines whether a translation is correct.
The reference is one possible translation, not the only valid one.

The previous analysis identified a broad translation problem.

Your task is to identify independent, minimal corrections
that are genuinely necessary to preserve the German meaning.

CHECK SEPARATELY:
1. Conjunctions and logical relationships
2. Missing or incorrectly added negation
3. Verb tense and aspect
4. Vocabulary and meaning
5. Grammar and prepositions
6. Missing or additional information

RULES:
- Return one correction per independent mistake.
- original_text must be an EXACT substring of the student translation.
- corrected_text must be a replacement for that substring.
- Keep corrections short but grammatically meaningful.
- Do not rewrite the entire sentence.
- Avoid overlapping original_text spans.
- Do not invent errors to reach a particular number.
- Accept valid paraphrases.
- Do not flag capitalization or numeric differences.
- If uncertain, omit the correction.

Use categories:
grammar, conjugation, plurals, omissions, word_order,
articles, prepositions, vocabulary, spelling, meaning,
terminology, other.

Return ONLY JSON:

{
  "mistakes": [
    {
      "category": "meaning",
      "original_text": "incorrect phrase",
      "corrected_text": "replacement phrase",
      "explanation": "Specific reason for the correction"
    }
  ]
}

If no reliable individual corrections exist:
{"mistakes": []}
"""

    return (
        instructions
        + "\n\nINPUT:\n"
        + json.dumps(data, ensure_ascii=False)
    )


def extract_individual_corrections(
    source,
    reference,
    prediction,
    feedback,
):
    """Ask local Qwen to decompose broad feedback."""

    from mlx_lm import load, generate

    model, tokenizer = load(
        "mlx-community/Qwen2.5-3B-Instruct-4bit"
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You identify precise translation mistakes. "
                "Return valid JSON only."
            ),
        },
        {
            "role": "user",
            "content": build_extraction_prompt(
                source, reference, prediction, feedback
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
