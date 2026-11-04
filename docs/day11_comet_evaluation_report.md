# Day 11 — COMET Evaluation Report

**Project:** TransistorAI  
**Date:** 2 November 2026  
**Language pair:** German → English (`de-en`)  
**Base model:** `Helsinki-NLP/opus-mt-de-en`  
**Neural evaluation metric:** `Unbabel/wmt22-comet-da`  
**Status:** Evaluation completed; final cross-system alignment verification and consolidated reporting to be confirmed.

---

## 1. Executive Summary

Day 11 extends the existing BLEU and chrF evaluation framework by introducing COMET, a neural machine translation evaluation metric.

Three translation systems were evaluated:

- **E0 — Pretrained baseline:** Original Helsinki-NLP MarianMT model without additional fine-tuning.
- **E1 — Technical fine-tuned:** Model fine-tuned exclusively on technical-domain data.
- **E2 — Mixed fine-tuned:** Model fine-tuned using technical and general-domain data.

The objective was to determine whether improvements measured by BLEU and chrF were also reflected in COMET, and to investigate the trade-off between technical-domain specialization and general-domain performance.

**Main findings:**

- E1 achieved the highest technical-domain scores across all three metrics.
- E0 retained the highest general-domain scores.
- E2 provided a stronger balance between technical specialization and general-domain performance than E1.
- COMET supported the overall performance patterns observed with BLEU and chrF.

These results suggest that domain-specific fine-tuning improves technical translation quality but can reduce general-domain performance. Mixed-domain fine-tuning partially mitigates this trade-off.

## 2. Evaluation Objectives

The Day 11 evaluation addressed four research questions:

1. Does COMET confirm the improvements measured by BLEU and chrF?
2. How does technical-only fine-tuning affect general-domain translation quality?
3. Does mixed-domain fine-tuning provide a better balance between domains?
4. Can the three systems be compared using identical held-out evaluation examples and reproducible settings?

## 3. Experimental Setup

### 3.1 Translation systems

| System | Description | Model |
|---|---|---|
| E0 | Original pretrained baseline | `Helsinki-NLP/opus-mt-de-en` |
| E1 | Technical-domain fine-tuning | `technical-finetuned/final_model` |
| E2 | Mixed-domain fine-tuning | `mixed-finetuned/final_model` |

All systems share the same pretrained model architecture.

### 3.2 Evaluation datasets

| Domain | Test examples |
|---|---:|
| Technical | 12,623 |
| General | 1,997 |
| **Total per system** | **14,620** |

The evaluation uses the held-out datasets prepared earlier in the project.

Dataset locations:

- `data/processed/technical/de-en/technical_test.csv`
- `data/processed/general/de-en/general_test.csv`

Each translation record contains the source sentence, reference translation, model prediction and sentence identifier.

The evaluation is designed to compare identical examples across systems. Final cross-system alignment verification should be documented after the independent validation script succeeds.

### 3.3 Translation generation settings

| Parameter | Value |
|---|---|
| Beam search | 4 beams |
| Sampling | Disabled |
| Maximum new tokens | 128 |
| Batch size | 8 |
| Random seed | 42 |
| Language direction | German → English |

E0 predictions were regenerated using the original Day 5 evaluation settings.

The regenerated BLEU and chrF scores matched the recorded Day 5 baseline to four decimal places.

## 4. Evaluation Metrics

### 4.1 BLEU

BLEU measures n-gram overlap between generated translations and reference translations.

Configuration:

- SacreBLEU
- Tokenization: `13a`

Higher scores indicate greater reference overlap.

### 4.2 chrF

chrF measures character n-gram similarity between generated and reference translations.

Configuration:

- Character order: 6
- Word order: 0
- Beta: 2

chrF is useful for assessing lexical and morphological similarity.

### 4.3 COMET

COMET is a learned machine translation evaluation metric that uses source sentences, candidate translations and reference translations to estimate translation quality.

**Model:** `Unbabel/wmt22-comet-da`

**Checkpoint revision:** `2760a223ac957f30acfb18c8aa649b01cf1d75f2`

COMET evaluation settings:

| Parameter | Value |
|---|---|
| Batch size | 8 |
| GPU count | 0 |
| Inference device | CPU |
| DataLoader workers | 1 |
| Evaluation mode | Full dataset |

The checkpoint was loaded from the Hugging Face cache. The revision above was observed in the checkpoint path; the evaluation script should be checked to confirm that it explicitly pins this revision.

COMET scores are reported as corpus-level system scores alongside sentence-level outputs.

## 5. Quantitative Results

### 5.1 Complete model comparison

| System | Domain | BLEU ↑ | chrF ↑ | COMET ↑ |
|---|---|---:|---:|---:|
| E0 | Technical | 35.1068 | 57.9039 | 0.8378 |
| E1 | Technical | **40.5126** | **61.8439** | **0.8676** |
| E2 | Technical | 39.8935 | 61.4829 | 0.8660 |
| E0 | General | **40.4083** | **65.4857** | **0.8668** |
| E1 | General | 30.6843 | 57.3025 | 0.8266 |
| E2 | General | 35.6534 | 61.6582 | 0.8493 |

Higher scores are better. COMET scores are rounded to four decimal places.

### 5.2 Technical-domain improvement over E0

| Model | Δ BLEU | Δ chrF | Δ COMET |
|---|---:|---:|---:|
| E1 | +5.4058 | +3.9400 | +0.0298 |
| E2 | +4.7867 | +3.5790 | +0.0282 |

Both fine-tuned systems improved technical-domain performance compared with the pretrained baseline.

E1 achieved the highest technical-domain scores, although the COMET difference between E1 and E2 was small.

### 5.3 General-domain change relative to E0

| Model | Δ BLEU | Δ chrF | Δ COMET |
|---|---:|---:|---:|
| E1 | −9.7240 | −8.1832 | −0.0402 |
| E2 | −4.7549 | −3.8273 | −0.0175 |

Both fine-tuned systems scored below E0 on the general-domain test set.

However, E2 experienced substantially smaller general-domain losses than E1.

### 5.4 E2 compared with E1

| Domain | Δ BLEU | Δ chrF | Δ COMET |
|---|---:|---:|---:|
| Technical | −0.6191 | −0.3610 | −0.0016 |
| General | +4.9691 | +4.3557 | +0.0227 |

E2 recovered general-domain performance while retaining nearly all of E1's technical-domain performance.

This suggests that mixed-domain fine-tuning can help reduce specialization-related performance losses.

## 6. Runtime and Computational Resources

COMET inference was executed on CPU, even though an Apple Silicon MPS device was available.

| System | Technical runtime | General runtime | Total |
|---|---:|---:|---:|
| E0 | 701.5 s | 150.1 s | 851.6 s |
| E1 | 638.1 s | 140.2 s | 778.3 s |
| E2 | 676.7 s | 147.5 s | 824.2 s |
| **Total** | **2,016.3 s** | **437.0 s** | **2,453.3 s** |

Total measured scoring runtime was approximately **40 minutes 53 seconds**, excluding model loading, translation generation and other overhead.

No peak RAM measurement was captured during these evaluations. Therefore, peak memory usage should be recorded as unavailable rather than estimated.

## 7. Reproducibility

### 7.1 Configurations

- `configs/metric_config.json`
- `configs/day11_comet_config.json`
- `configs/day11_e0_comet_config.json`
- `configs/day11_e0_eval_config.json`

### 7.2 Evaluation scripts

- `evaluation/evaluate_finetuned.py`
- `evaluation/day11_evaluate_comet.py`

### 7.3 Evaluation artifacts

E0 translation predictions:

`results/day11_evaluation/e0_generation/predictions.csv`

E0 COMET outputs:

`results/day11_evaluation/e0_comet/`

E1 and E2 COMET outputs:

`results/day11_evaluation/`

Consolidated comparison (after successful generation):

`results/day11_evaluation/e0_e1_e2_comet_comparison.csv`

### 7.4 Environment and version tracking

The evaluation environment uses Python 3.11 and PyTorch Lightning with the COMET package.

The exact package versions should be taken from the saved environment metadata, rather than inferred from warning messages.

The evaluation produced nonfatal warnings related to deprecated APIs, checkpoint compatibility and an unused checkpoint key. All three systems completed successfully.

## 8. Interpretation

### 8.1 Technical specialization

Technical-domain fine-tuning produced improvements over the pretrained baseline across BLEU, chrF and COMET.

This indicates that the technical training data improved the models' performance on the selected technical evaluation set.

### 8.2 General-domain performance loss

E1 exhibited a substantial reduction in general-domain scores.

This is consistent with a domain-specialization trade-off: improvements on technical text were accompanied by reduced performance on the general test set.

The results do not establish the exact cause of the decline, so it should not automatically be described as proven catastrophic forgetting.

### 8.3 Benefits of mixed-domain fine-tuning

E2 reduced the general-domain performance loss while retaining a strong technical-domain score.

This makes E2 a promising candidate when the translation system must handle both technical and general text.

### 8.4 Agreement between metrics

BLEU, chrF and COMET agreed on the ranking of systems within both domains:

**Technical:** E1 > E2 > E0

**General:** E0 > E2 > E1

This agreement strengthens the descriptive evidence for the observed domain trade-off.

## 9. Limitations

Several limitations should be considered when interpreting the results:

- COMET is a learned evaluation metric, not a substitute for human evaluation.
- The results are specific to the selected technical and general-domain test sets.
- Small score differences, particularly E1 versus E2 technical COMET, require statistical significance testing.
- The independent cross-model alignment check must be confirmed before claiming identical examples were verified.
- CPU evaluation time was recorded, but peak memory usage was not.
- The COMET checkpoint revision should be verified as explicitly pinned for reproducibility.
- COMET scores were calculated using the available references; reference quality can affect evaluation.

## 10. Conclusion

Day 11 successfully integrated COMET into the TransistorAI evaluation framework and produced neural evaluation scores for E0, E1 and E2.

The results demonstrate a clear trade-off between technical specialization and general-domain translation quality.

E1 achieved the strongest technical-domain results, while E0 retained the strongest general-domain performance.

E2 provided the best observed balance among the two fine-tuned systems, improving general-domain performance relative to E1 with only a small reduction in technical performance.

**Recommended model selection:**

- **E1:** When technical-domain performance is the primary objective.
- **E0:** When general-domain performance is the primary objective.
- **E2:** When a single fine-tuned model must support both domains.

Future work should investigate statistical significance, targeted error analysis and deployment-oriented evaluation.

---

## 11. Day 11 Completion Checklist

- [x] Regenerate E0 predictions using the original baseline configuration.
- [x] Reproduce E0 BLEU and chrF scores.
- [x] Evaluate E0, E1 and E2 with COMET.
- [x] Record full-dataset scores and evaluation runtimes.
- [x] Compare technical and general-domain performance.
- [X] Confirm independent cross-model sentence alignment.
- [X] Verify and document exact dependency versions and checkpoint pinning.
- [X] Confirm the consolidated comparison CSV exists.
- [X] Commit the final report and reproducibility artifacts to Git.