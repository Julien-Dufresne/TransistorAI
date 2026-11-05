# TransistorAI — Day 14 Report
## LLM Baseline and Four-System Translation Evaluation

**Project:** TransistorAI — German-to-English Machine Translation  
**Day:** 14  
**Date:** 5 November 2026  
**Status:** Evaluation completed; reporting and reproducibility review

---

## 1. Executive Summary

Day 14 focused on extending TransistorAI's neural machine translation (NMT) experiments with a large language model (LLM) baseline and comparing four translation systems across technical and general-language domains.

The four systems evaluated were:

- **E0 — Baseline NMT:** Helsinki-NLP/opus-mt-de-en, without project-specific fine-tuning.
- **E1 — Technical NMT:** Fine-tuned primarily for technical translation.
- **E2 — Mixed NMT:** Fine-tuned using a mixture of technical and general-language data.
- **E3 — LLM baseline:** Qwen 2.5 3B Instruct, used for German-to-English translation through MLX-based inference.

All systems were compared on the same domain-specific evaluation sets, comprising **14,620 sentences** in total.

The evaluation used three complementary metrics: BLEU, chrF and COMET.

### Main findings

1. **E1 achieved the highest scores on technical translation**, outperforming the original baseline and the other fine-tuned or LLM-based systems.
2. **E0 remained the strongest system for general-language translation**, illustrating the cost of domain-specific fine-tuning.
3. **E2 offered a compromise between technical specialization and general-language performance**, although it did not lead either domain.
4. **E3 did not outperform the NMT systems on the automated metrics.** Nevertheless, its general-domain COMET score was relatively close to E0, making it a useful candidate for subsequent human preference evaluation.

The results support the value of domain-specific fine-tuning while motivating further investigation into whether automated metrics align with human judgments.

---

## 2. Objectives

The objectives of Day 14 were to:

- Introduce an LLM-based translation baseline alongside the existing NMT models.
- Generate German-to-English translations for the technical and general evaluation sets.
- Validate prediction completeness and alignment.
- Compare all four systems using BLEU, chrF and COMET.
- Identify performance differences between technical and general translation.
- Prepare the results for integration into the Streamlit MVP and subsequent human evaluation.

The purpose of E3 was to establish an additional architectural baseline, not necessarily to surpass the fine-tuned NMT systems.

---

## 3. Models and Experimental Setup

| System | Model / approach | Training or adaptation |
|---|---|---|
| E0 | Helsinki-NLP/opus-mt-de-en | Original pretrained baseline |
| E1 | Helsinki-NLP/opus-mt-de-en | Technical-domain fine-tuning |
| E2 | Helsinki-NLP/opus-mt-de-en | Mixed technical/general fine-tuning |
| E3 | Qwen 2.5 3B Instruct | Prompt-based translation baseline |

### 3.1 NMT systems

The E0–E2 systems share the same underlying sequence-to-sequence translation architecture, allowing the project to examine how different fine-tuning strategies affect translation quality.

E1 emphasizes technical-language adaptation. E2 combines technical and general-language training examples to reduce the loss of general translation capability.

### 3.2 LLM system

E3 uses Qwen 2.5 3B Instruct as a general-purpose instruction-following language model for translation.

The implementation uses MLX tooling on Apple Silicon. The project used a 4-bit Qwen configuration for local inference.

Unlike E1 and E2, E3 was evaluated as a prompted LLM baseline rather than a model fine-tuned specifically for the TransistorAI datasets.

The exact prompt, decoding configuration, checkpoint revision and inference parameters should be preserved in the generation script and recorded in the reproducibility documentation.

---

## 4. Evaluation Datasets

Two held-out German-to-English test sets were used.

| Dataset | Domain | Test sentences |
|---|---|---:|
| KDE4 | Technical/software localization | 12,623 |
| Tatoeba/OPUS | General language | 1,997 |
| **Total** | **Both domains** | **14,620** |

The technical dataset contains software-related language, including interface messages and technical terminology.

The general dataset provides a complementary measure of broader translation performance.

### Evaluation alignment

For a fair comparison, model outputs must correspond to the same source sentences and references.

The evaluation pipeline was designed to preserve sentence-level alignment across E0, E1, E2 and E3.

The Day 14 workflow also included a recovery procedure for truncated LLM outputs. These recovery operations should be documented because output completeness can affect aggregate metric scores.

---

## 5. Evaluation Metrics

Three automatic machine translation metrics were used.

### 5.1 BLEU

BLEU measures the overlap of word sequences between a generated translation and a reference translation.

Higher scores indicate stronger correspondence with reference wording.

**Limitation:** BLEU may penalize valid translations that use alternative phrasing.

### 5.2 chrF

chrF measures character-level n-gram similarity between generated and reference translations.

It is useful for identifying similarities involving morphology, spelling and word forms.

**Limitation:** High character overlap does not necessarily guarantee that the translated meaning is correct.

### 5.3 COMET

COMET is a learned translation-quality metric that evaluates semantic and contextual relationships among source text, candidate translation and reference translation.

The project uses **Unbabel/wmt22-comet-da**.

COMET provides a complementary perspective to lexical-overlap metrics.

**Limitation:** COMET remains an automated model-based assessment and can disagree with human judgments.

Together, the three metrics offer a broader assessment than any single score.

---

## 6. Quantitative Results

### 6.1 Technical domain

**Dataset:** KDE4  
**Number of sentences:** 12,623

| Model | BLEU ↑ | chrF ↑ | COMET ↑ |
|---|---:|---:|---:|
| E0 — Baseline | 35.1068 | 57.9039 | 0.8378 |
| **E1 — Technical** | **40.5126** | **61.8439** | **0.8676** |
| E2 — Mixed | 39.8935 | 61.4829 | 0.8660 |
| E3 — Qwen | 29.1308 | 54.0144 | 0.8112 |

E1 was the strongest technical translation system across all three metrics.

Relative to E0, E1 improved:

- BLEU by approximately **5.41 points**.
- chrF by approximately **3.94 points**.
- COMET by approximately **0.0298**.

E2 performed close to E1, suggesting that mixed-domain fine-tuning preserved much of the benefit of technical adaptation.

E3 scored below all three NMT systems on this dataset.

### 6.2 General domain

**Dataset:** Tatoeba/OPUS  
**Number of sentences:** 1,997

| Model | BLEU ↑ | chrF ↑ | COMET ↑ |
|---|---:|---:|---:|
| **E0 — Baseline** | **40.4083** | **65.4857** | **0.8668** |
| E1 — Technical | 30.6843 | 57.3025 | 0.8266 |
| E2 — Mixed | 35.6534 | 61.6582 | 0.8493 |
| E3 — Qwen | 33.5082 | 61.1845 | 0.8584 |

E0 achieved the highest general-domain scores across all three metrics.

E1 experienced a substantial decline in general-domain performance compared with E0.

E2 partially recovered general-language performance relative to E1.

E3 produced a lower BLEU and chrF than E2, but a higher COMET score, suggesting that lexical and learned semantic metrics did not rank these two systems identically.

### 6.3 Cross-domain comparison

| Model | Technical BLEU | General BLEU | Technical COMET | General COMET |
|---|---:|---:|---:|---:|
| E0 | 35.1068 | 40.4083 | 0.8378 | 0.8668 |
| E1 | 40.5126 | 30.6843 | 0.8676 | 0.8266 |
| E2 | 39.8935 | 35.6534 | 0.8660 | 0.8493 |
| E3 | 29.1308 | 33.5082 | 0.8112 | 0.8584 |

These results show that the relative ranking of translation systems depends on the domain and metric.

Cross-domain score differences should not be interpreted as pure measures of domain difficulty, because the datasets differ in their sentence distributions, reference styles and content.

---

## 7. Findings and Interpretation

### 7.1 Technical fine-tuning was effective

E1 achieved the highest technical-domain performance.

This supports the hypothesis that adapting a pretrained NMT system to software-related language can improve translation quality within the target domain.

However, the improvement should be interpreted alongside E