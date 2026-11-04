from sacrebleu.metrics import BLEU, CHRF

bleu_metric = BLEU(effective_order=True)
chrf_metric = CHRF(word_order=0)


def calculate_translation_metrics(
    hypothesis: str,
    reference: str,
) -> dict:
    """
    Calculate sentence-level BLEU and chrF.
    Both metrics return scores from 0 to 100.
    """
    if not hypothesis.strip() or not reference.strip():
        raise ValueError(
            "Hypothesis and reference must not be empty."
        )

    bleu = bleu_metric.sentence_score(
        hypothesis,
        [reference],
    ).score

    chrf = chrf_metric.sentence_score(
        hypothesis,
        [reference],
    ).score

    return {
        "bleu": round(bleu, 2),
        "chrf": round(chrf, 2),
    }
