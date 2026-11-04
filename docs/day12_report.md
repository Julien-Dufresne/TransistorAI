Today's objective:
To build a reusable evalution module that checks whether machine-translated text preserves important technical terminology and formatting. 
The existing evaluation pipeline measures translation quality using BLEU, chrF and COMET. Today we will introduce a 
**rule-based validation** to detect specific errors that these metrics may not identify reliably. 

We will evaluate: 
- Technical terminology (Ex. 'Leiterplatte' -> 'printed circuit board')
- Product name (ex. 'Siemens S7-1200')
- Numbers (ex. '24')
- Units (ex. '24 V')
- Placeholders (ex. '{device_id})
- Formatting tokens (ex. '%s' or '<code>')


# Day 12 — Technical Terminology and Protected-Content Evaluation

**Project:** TransistorAI  
**Translation direction:** German → English  
**Models:** E0 (pretrained baseline), E1 (technical fine-tuned), E2 (mixed-domain fine-tuned)  
**Evaluation:** Rule-based checks on aligned technical test examples

## 1. Objective

The objective of Day 12 was to evaluate technical translation reliability beyond aggregate BLEU and chrF scores.

Six dimensions were evaluated:

- Technical terminology
- Numerical preservation
- Placeholder preservation
- Technical units
- URL preservation
- File-path preservation

All three models were evaluated using aligned source sentences and reference translations.

## 2. Methodology

The evaluation pipeline uses `evaluation/day12_evaluate_terminology.py` and the check functions implemented in `evaluation/day12_terminology_checks.py`.

Each check is applied only when the source sentence contains the relevant feature.

Results are aggregated by model, domain and evaluation category.

The evaluation uses strict, rule-based compliance criteria. These metrics complement translation-quality metrics but do not replace human assessment.

## 3. Quantitative Results

| Metric | Applicable checks | E0 | E1 | E2 |
|---|---:|---:|---:|---:|
| Terminology | 971 | 98.15% | 93.51% | 94.85% |
| Numbers | 864 | 98.15% | 91.32% | 91.20% |
| Placeholders | 896 | 99.89% | 99.89% | 99.89% |
| Units | 25 | 100.00% | 100.00% | 100.00% |
| URLs | 69 | 84.06% | 0.00% | 0.00% |
| File paths | 85 | 88.24% | 0.00% | 0.00% |

### 3.1 Terminology

E0 achieved the highest terminology compliance at 98.15%.

E1 achieved 93.51%, representing a decrease of 4.64 percentage points compared with E0.

E2 achieved 94.85%, representing a decrease of 3.30 percentage points.

The mixed-domain model performed slightly better than the technical-only model on this glossary-based metric.

### 3.2 Numerical preservation

E0 preserved numerical content in 848 of 864 applicable checks.

E1 passed 789 checks and E2 passed 788 checks.

Both fine-tuned models therefore exhibited reduced numerical preservation relative to the baseline.

### 3.3 Placeholder preservation

All three models achieved 895 successful checks out of 896.

Placeholder preservation remained stable across the experiments.

### 3.4 Technical units

All models passed 25 of 25 applicable unit checks.

However, this is a relatively small evaluation subset, and the detector covers only recognized unit patterns. The result does not establish perfect unit handling in unrestricted technical text.

### 3.5 URLs

E0 preserved 58 of 69 URL-containing examples exactly.

E1 and E2 passed none of the 69 strict URL-preservation checks.

Controlled generation experiments also revealed whitespace insertion inside URLs in fine-tuned model outputs.

### 3.6 File paths

E0 preserved 75 of 85 file-path-containing examples exactly.

E1 and E2 passed none of the 85 strict file-path-preservation checks.

Observed errors included inserted spaces around directory separators and file extensions.

## 4. Qualitative Error Analysis

Manual review of the baseline's 21 protected-content failures identified several categories.

URL-related differences included trailing-slash changes, translation truncation, changes to URL content and potentially intentional localization.

File-path differences included whitespace corruption, truncation and translated or otherwise modified path components.

A strict exact-match failure is not necessarily equivalent to a functional failure. For example, adding a trailing slash may preserve URL usability, whereas inserting whitespace into a URL may make it unusable without repair.

The fine-tuned models showed systematic exact-preservation failures in the evaluated protected-content examples.

## 5. Interpretation

The Day 12 evaluation demonstrates that aggregate translation-quality metrics do not fully capture technical reliability.

Fine-tuning can improve reference-based translation similarity while simultaneously reducing the preservation of technical strings.

This is particularly relevant to software documentation, where URLs, filesystem paths, numbers and technical terminology may require exact reproduction.

E0 is the strongest model on the evaluated exact-preservation dimensions. However, these results alone do not determine the best overall translation model.

## 6. Limitations

- The evaluation is based on rule-based checks rather than comprehensive human judgments.
- Exact-string compliance does not directly measure URL or path usability.
- Glossary matching may penalize valid terminology alternatives.
- The unit detector recognizes a limited set of patterns.
- The file-path detector intentionally excludes certain ambiguous path formats.
- The URL detector may include surrounding punctuation in some cases.
- Compliance percentages are descriptive results for the evaluated test examples and should not be interpreted as universal model reliability.
- The observed regression is associated with fine-tuned generation outputs; its precise training-related mechanism has not been established.

## 7. Recommendations

1. Preserve protected URLs and filesystem paths using reversible placeholders before translation, then restore them afterward.
2. Add automated integrity checks for protected technical strings.
3. Evaluate the impact of protected-token preprocessing on BLEU, chrF and technical compliance.
4. Investigate number preservation separately, including punctuation and formatting differences.
5. Expand unit tests and glossary coverage.
6. Consider additional controlled fine-tuning only after testing preprocessing and decoding-level mitig