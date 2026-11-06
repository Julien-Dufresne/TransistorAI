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

However, the improvement should be interpreted alongside E1's decline on the general test set.

### 7.2 Domain specialization introduced a trade-off

E1's general BLEU declined by approximately 9.72 points relative to E0.

This is consistent with a domain adaptation trade-off: improving performance on specialized content can reduce performance on content outside that domain.

The current evaluation demonstrates the trade-off but does not establish its precise cause. Additional experiments would be needed to distinguish catastrophic forgetting from other effects of fine-tuning and dataset differences.

### 7.3 Mixed-domain training reduced the trade-off

E2 retained strong technical-domain performance while outperforming E1 on the general-domain test set.

Relative to E1:

- Technical BLEU decreased by approximately 0.62 points.
- General BLEU increased by approximately 4.97 points.

E2 is therefore a practical candidate when one translation system must support both domains.

### 7.4 The LLM baseline did not dominate

E3 achieved lower BLEU and chrF than E0 on both domains.

It also achieved the lowest technical COMET score.

On the general dataset, however, E3's COMET score of 0.8584 was relatively close to E0's 0.8668.

This makes E3 relevant for human evaluation despite its weaker lexical-overlap results.

The results do not establish that LLMs are generally inferior to NMT. They compare specific models, adaptation strategies, inference configurations and datasets.

### 7.5 Automated metrics do not provide a complete quality assessment

The different rankings of E2 and E3 on the general dataset illustrate why multiple metrics are useful.

A translation can differ substantially from a reference while still preserving meaning, or appear similar to the reference while introducing an important semantic error.

A human evaluation is needed to examine preferences that automated metrics may not fully capture.

---

## 8. Limitations

### 8.1 Model size and architecture are not controlled

The comparison includes specialized NMT models and a general-purpose instruction-tuned LLM.

Differences in architecture, parameter count, pretraining and adaptation mean that this is a practical system comparison rather than a controlled architectural experiment.

### 8.2 Prompt and decoding sensitivity

LLM translation performance may depend on prompt wording, generation settings, quantization and output-length limits.

The present results describe the tested E3 configuration, not every possible Qwen configuration.

### 8.3 Reference-based metric limitations

BLEU and chrF rely on reference overlap.

COMET offers a learned quality estimate but can still produce incorrect rankings.

None of these metrics independently guarantees the preservation of meaning, terminology, negation, numbers or placeholders.

### 8.4 Dataset representativeness

KDE4 and Tatoeba/OPUS are useful evaluation datasets, but they do not represent all possible professional translation scenarios.

Results may differ on longer documents, modern technical documentation, specialized terminology or conversational text.

### 8.5 Unequal domain sizes

The technical test set contains substantially more sentences than the general test set.

Domain-specific results should therefore be reported separately rather than interpreted through an unqualified combined score.

### 8.6 Human preferences have not yet been incorporated

The Day 14 conclusions are based on automatic evaluation.

Human evaluation may reveal differences in naturalness, readability and perceived adequacy that are not captured by the aggregate scores.

### 8.7 Statistical significance has not been established

The reported score differences have not been accompanied here by confidence intervals or significance tests.

Small differences, especially between E1 and E2 on the technical domain, should not automatically be treated as statistically significant.

---

## 9. Reproducibility and Artifacts

The primary Day 14 artifacts are located in:

`results/day14_llm_baseline/`

The aggregate comparison is stored in:

`results/day14_llm_baseline/full_model_comparison.csv`

Relevant evaluation scripts include:

- `evaluation/day14_generate_llm.py`
- `evaluation/day14_recover_truncations.py`
- `evaluation/day14_compare_pilot.py`
- `evaluation/day14_evaluate_comet_pilot.py`
- `evaluation/day14_evaluate_comet_full.py`

### Reproducibility considerations

The final research record should preserve:

1. Model identifiers and revisions.
2. Dataset versions and evaluation split definitions.
3. The E3 translation prompt.
4. Quantization and generation settings.
5. Sentence alignment and recovery procedures.
6. BLEU, chrF and COMET implementation details.
7. Software versions and execution environment.
8. Final prediction counts and evaluation outputs.

The exact commands and parameter values should be taken from the corresponding project scripts rather than reconstructed from memory.

---

## 10. Implications for the TransistorAI MVP

The four-model comparison provides the quantitative foundation for the next stage of the project.

The Streamlit MVP can present multiple translations for the same German source sentence, allowing users to compare systems directly.

The interface should hide model identities during voting to reduce potential model-name bias.

Human selections can then be compared with BLEU, chrF and COMET rankings.

This creates a link between automated machine translation evaluation and human-centered assessment.

---

## 11. Next Steps

### Day 15 — Human Preference Evaluation

Implement and validate a Streamlit interface containing 200 German source sentences, with 100 technical and 100 general examples.

For each sentence:

- Display anonymized translations from E0, E1, E2 and E3.
- Randomize translation presentation order.
- Group identical translations to avoid duplicate options.
- Store participant selections in SQLite.
- Support participant identification and evaluation resumption.
- Track evaluation progress independently for each domain.

### Subsequent evaluation

Use the collected votes to compare human preferences against automated metric rankings.

Investigate cases in which human choices disagree with BLEU, chrF or COMET.

Where feasible, report agreement statistics and analyze differences by domain, model and error category.

### Final MVP and presentation

Integrate the findings into the Streamlit application and final presentation, emphasizing:

- The benefit of technical-domain adaptation.
- The general-language trade-off.
- The role of mixed-domain fine-tuning.
- The value and limitations of the LLM baseline.
- The relationship between automated metrics and human preferences.

---

## 12. Conclusion

Day 14 established a four-system comparison framework for TransistorAI.

The results demonstrate that technical fine-tuning can substantially improve translation quality on the target domain, but that specialization may reduce general-language performance.

Mixed-domain fine-tuning provided a more balanced alternative, while the Qwen LLM baseline offered a useful comparison against the NMT systems.

The most important conclusion is that **translation quality depends on the target domain, model adaptation and evaluation method**.

The next stage will extend this quantitative evaluation with human preference data, helping determine whether users select the same translations that automated metrics rank highest.

**Day 14 outcome:** Four-system automatic evaluation completed; human-centered evaluation is the next research milestone.