# TransistorAI — Day 9–10 Evaluation Report

## 1. Executive Summary

This report evaluates the impact of domain-specific and mixed-domain fine-tuning on German-to-English machine translation using MarianMT.

Three experiments were compared:

- **E0:** Pretrained MarianMT baseline.
- **E1:** Fine-tuned exclusively on technical translation data.
- **E2:** Fine-tuned on a mixture of technical and general-domain data.

The experiments were evaluated using SacreBLEU BLEU, chrF, and a qualitative review of 50 translation examples.

**Key finding:** Technical-only fine-tuning achieved the highest technical BLEU score, while mixed-domain fine-tuning provided a better balance between technical-domain improvement and general-domain retention.

## 2. Experimental Setup

### Model

Helsinki-NLP/opus-mt-de-en (MarianMT)

### Translation Direction

German → English

### Evaluation Domains

- Technical: KDE4
- General: Tatoeba/OPUS

### Fine-Tuning Configuration

- Learning rate: 2e-5
- Epochs: 3
- Training batch size: 4
- Gradient accumulation: 2
- Effective batch size: 8
- Random seed: 42

E1 and E2 were intended to use the same pretrained checkpoint and comparable training settings. The exact training budgets and dataset compositions should be verified from the saved experiment configurations before claiming a fully controlled comparison.

## 3. Quantitative Results

| Experiment | Technical BLEU | Technical chrF | General BLEU | General chrF |
|---|---:|---:|---:|---:|
| E0 | 35.1068 | 57.9039 | 40.4074 | 65.4856 |
| E1 | 40.5126 | 61.8439 | 30.6843 | 57.3025 |
| E2 | 39.8935 | 61.4829 | 35.6534 | 61.6582 |

### Technical Domain

E1 improved technical BLEU by 5.41 points over E0.

E2 improved technical BLEU by 4.79 points over E0.

The difference between E1 and E2 was approximately 0.62 BLEU points in favor of E1.

### General Domain

E1 lost approximately 9.72 BLEU points relative to E0.

E2 lost approximately 4.75 BLEU points relative to E0.

Compared with E1, E2 recovered approximately 4.97 BLEU points on general-domain data.

## 4. Qualitative Evaluation

A sample of 50 translations was manually reviewed:

- 25 technical translations.
- 25 general-domain translations.

The review considered terminology, mistranslation, omission, grammar, fluency, hallucination, error severity and meaning preservation.

### Human Model Preferences

| Preference | Count |
|---|---:|
| E1 | 6 |
| E2 | 11 |
| Tie | 33 |

### Error Severity

| Severity | E1 | E2 |
|---|---:|---:|
| None | 15 | 20 |
| Minor | 12 | 10 |
| Moderate | 13 | 10 |
| Major | 10 | 10 |

These figures describe the selected review sample. They do not establish a statistically significant difference.

### Observations

Both models generally preserved straightforward technical instructions and common interface terminology.

More difficult cases involved domain-specific terms, named entities, numbers, idiomatic expressions and source-code preservation.

E2 produced more error-free translations in the reviewed sample, although both models had the same number of major-error cases.

## 5. Discussion

The results demonstrate a trade-off between technical specialization and general-domain retention.

E1 maximized technical-domain BLEU but experienced a substantial decline in general-domain performance.

E2 retained most of E1's technical improvement while reducing the general-domain performance loss.

The qualitative review broadly supports this interpretation, but its deliberately selected sample cannot establish population-level superiority.

## 6. Limitations

1. The qualitative sample contains only 50 translations and was enriched for model disagreements.
2. Human annotations are provisional and have not undergone independent inter-annotator agreement testing.
3. Reference translations may contain ambiguities or errors.
4. Differences in training-data composition and total optimization steps may affect the fairness of the E1/E2 comparison.
5. No confidence intervals or paired significance tests have yet been calculated.
6. BLEU and chrF do not fully capture semantic correctness or technical usability.

## 7. Conclusion

Technical-only fine-tuning is appropriate when maximizing performance on the technical test domain is the primary objective.

Mixed-domain fine-tuning appears more suitable when both technical quality and general-domain retention matter.

For TransistorAI, E2 is the preferred candidate for further balanced-domain evaluation, while E1 remains the technical-domain performance leader.

## 8. Next Steps

- Review the qualitative annotations and investigate severe errors.
- Run paired statistical significance tests or bootstrap confidence intervals.
- Verify training-budget comparability.
- Perform a targeted regression analysis on technical terminology, numbers and code snippets.
- Evaluate model performance on additional out-of-domain datasets.
