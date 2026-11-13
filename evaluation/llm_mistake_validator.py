
"""Second-pass validation of Qwen translation suggestions."""

import json
import re


def validate_llm_mistake(
    source,
    reference,
    prediction,
    mistake,
    return_details=False,
):
    """
    Validate a proposed translation mistake.

    Returns:
        True: accepted
        False: rejected
        None: uncertain or failed

    Set return_details=True to inspect the model's reasoning.
    """

    from mlx_lm import load, generate

    model_id = "mlx-community/Qwen2.5-3B-Instruct-4bit"
    model, tokenizer = load(model_id)

    payload = {
        "german_source": source,
        "reference_translation": reference,
        "student_translation": prediction,
        "proposed_mistake": {
            "category": mistake["category"],
            "original_text": mistake["original_text"],
            "corrected_text": mistake["corrected_text"],
            "explanation": mistake["explanation"],
        },
    }

    
    
    system_prompt = """
    You are an evaluator of German-to-English translation errors.

    Your job is to judge ONE proposed correction to a student's
    English translation.

    The German source is authoritative.
    The reference is only one possible correct translation.

    Evaluate these questions in order:

    1. Does the student's original English wording accurately
    preserve the relevant meaning of the German source?

    2. Does the proposed corrected wording fix a genuine error
    in the student's translation?

    3. Is the difference merely a natural paraphrase, contraction,
    politeness variation, or stylistic preference?

    DECISION RULES:

    ACCEPT:
    - The student's original wording contains a genuine
    translation or grammatical error.
    - The proposed correction fixes that error.
    - Other unrelated errors may remain.

    REJECT:
    - The student's original wording is already acceptable.
    - The correction is only a stylistic preference.
    - The correction introduces a new error.
    - The proposed correction does not fix the identified error.

    UNCERTAIN:
    - The evidence is insufficient to determine correctness.

    IMPORTANT:
    - Never correct a valid paraphrase just because it differs
    from the reference.
    - Contractions such as "I'm" and "I am" are equivalent.
    - "Could you help me?" is an acceptable polite rendering
    of "Kannst du mir helfen?"
    - Missing negation reverses meaning.
    - Opposite meanings such as "allowed" and "forbidden"
    are genuine translation errors.
    - A source-language noun must not be replaced with a
    different object merely because it fits the situation.
    - Do not evaluate the German source as if it were
    the student's English translation.
    - Do not assume the proposed explanation is correct.

    Return exactly one JSON object containing:
    - decision: accept, reject, or uncertain
    - reason: a specific explanation mentioning the actual
    German and English words involved.

    Do not include Markdown, examples, or additional text.
    """



    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=False,
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
        max_tokens=256,
        verbose=False,
    )

    
    
    match = re.search(r"\{[\s\S]*\}", response)

    details = {
        "decision": "uncertain",
        "reason": "",
        "raw_response": response,
    }

    if match:
        try:
            parsed = json.loads(match.group())

            if isinstance(parsed, dict):
                decision = parsed.get("decision", "uncertain")
                reason = parsed.get("reason", "")

                # Validate the explanation.
                if (
                    not isinstance(reason, str)
                    or len(reason.strip()) < 15
                    or reason.strip().lower()
                    == "brief explanation grounded in the german source"
                ):
                    decision = "uncertain"
                    reason = reason if isinstance(reason, str) else ""

                # Check whether Qwen's decision contradicts
                # its own explanation.
                decision = check_decision_consistency(
                    decision,
                    reason,
                )

                if decision in {
                    "accept",
                    "reject",
                    "uncertain",
                }:
                    details["decision"] = decision

                details["reason"] = reason

        except json.JSONDecodeError:
            pass


    decision_map = {
        "accept": True,
        "reject": False,
        "uncertain": None,
    }

    result = decision_map[details["decision"]]

    if return_details:
        return {
            "result": result,
            **details,
        }

    return result





def check_decision_consistency(decision, reason):
    """
    Flag contradictory validator decisions while avoiding
    suppression of explicitly identified translation errors.
    """

    if not isinstance(reason, str):
        return "uncertain"

    text = reason.casefold()

    strong_acceptability = (
        "natural paraphrase",
        "only a stylistic preference",
        "no correction is necessary",
        "already grammatically correct",
        "despite being a contraction",
    )

    explicit_error = (
        "omits",
        "omitted",
        "missing",
        "adding",
        "added",
        "error of",
        "genuine error",
        "genuine translation error",
        "fixes the identified error",
        "fixes a genuine error",
        "incorrect tense",
        "grammatical error",
    )

    says_acceptable = any(
        phrase in text for phrase in strong_acceptability
    )

    identifies_error = any(
        phrase in text for phrase in explicit_error
    )

    if decision == "accept":
        if says_acceptable and not identifies_error:
            return "uncertain"

        if (
            "accurately preserves the meaning" in text
            or "accurately preserves the relevant meaning" in text
        ) and not identifies_error:
            return "uncertain"

    return decision


