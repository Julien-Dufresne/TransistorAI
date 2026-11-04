# TransistorAI — Day 13 Error Analysis Report

**Language pair:** German → English  
**Models:** E0 baseline, E1 technical fine-tuned, E2 mixed fine-tuned  
**Date:** 2026-11-04

## 1. Executive Summary

This evaluation examines 300 model outputs from 100 sampled
German source sentences.

- Total outputs: 300
- Manually reviewed outputs: 273
- Outputs requiring data-quality review: 27
- Pending outputs: 0

The annotation dataset uses one primary error category per output.

The sample includes randomly selected and deliberately high-risk
technical sentences. Consequently, the observed error rates
should not be interpreted as unbiased corpus-wide error rates.

## 2. Model Comparison

| model   |   reviewed_outputs |   no_error_outputs |   error_outputs |   error_rate_percent |   mean_comet |
|:--------|-------------------:|-------------------:|----------------:|---------------------:|-------------:|
| E0      |                 91 |                 62 |              29 |              31.8681 |       0.8555 |
| E1      |                 91 |                 52 |              39 |              42.8571 |       0.8534 |
| E2      |                 91 |                 56 |              35 |              38.4615 |       0.8553 |

E0 has the lowest annotated error rate in this reviewed sample.

E1 has the highest error rate, while E2 falls between E0 and E1.

These observations describe the manually reviewed sample rather
than establishing overall model superiority.

## 3. Error Taxonomy

Primary categories:

- Semantic
- Omission
- Terminology
- Grammar
- Formatting
- No error

Error severity levels:

- Minor
- Major
- Critical

### Category Counts

| model   |   formatting |   grammar |   no_error |   omission |   semantic |   terminology |
|:--------|-------------:|----------:|-----------:|-----------:|-----------:|--------------:|
| E0      |            1 |         7 |         62 |          4 |          5 |            12 |
| E1      |            3 |         4 |         52 |          6 |         17 |             9 |
| E2      |            4 |         8 |         56 |          2 |         13 |             8 |

## 4. Severity Analysis

| model   |   minor |   major |   critical |
|:--------|--------:|--------:|-----------:|
| E0      |      20 |       8 |          1 |
| E1      |      17 |      21 |          1 |
| E2      |      19 |      15 |          1 |

Major semantic errors are especially important because they can
change the intended meaning of translated instructions.

## 5. Domain-Specific Analysis

| domain    | model   |   reviewed_outputs |   error_outputs |   error_rate_percent |
|:----------|:--------|-------------------:|----------------:|---------------------:|
| general   | E0      |                 19 |               9 |              47.3684 |
| general   | E1      |                 19 |              16 |              84.2105 |
| general   | E2      |                 19 |              11 |              57.8947 |
| technical | E0      |                 72 |              20 |              27.7778 |
| technical | E1      |                 72 |              23 |              31.9444 |
| technical | E2      |                 72 |              24 |              33.3333 |

The general-domain results suggest that technical specialization
may reduce general-domain translation quality.

However, this sample is relatively small and stratified.
Additional representative evaluation is needed to quantify
domain-wide effects.

## 6. COMET Versus Human Judgment

| model   | has_error   |   count |   mean_comet |   median_comet |
|:--------|:------------|--------:|-------------:|---------------:|
| E0      | False       |      62 |       0.8703 |         0.8748 |
| E0      | True        |      29 |       0.8237 |         0.8281 |
| E1      | False       |      52 |       0.884  |         0.8818 |
| E1      | True        |      39 |       0.8125 |         0.8309 |
| E2      | False       |      56 |       0.8827 |         0.8991 |
| E2      | True        |      35 |       0.8113 |         0.8363 |

COMET generally assigns lower scores to translations that human
reviewers classified as erroneous.

Nevertheless, COMET is not a guarantee that a translation
preserves critical instructions or negation.

## 7. Critical Errors

| sample_id     | model   | domain    | error_category   | error_subtype   |   comet_score |
|:--------------|:--------|:----------|:-----------------|:----------------|--------------:|
| technical_609 | E0      | technical | semantic         | lost_negation   |        0.8309 |
| technical_609 | E1      | technical | semantic         | lost_negation   |        0.8148 |
| technical_609 | E2      | technical | semantic         | lost_negation   |        0.8207 |

### Lost Negation: technical_609

All three models incorrectly translate the instruction to
uncheck a user-interface option as an instruction to select it.

This reverses the required user action.

E1 and E2 also introduce a formatting error in the filename
index.docbook.

The finding demonstrates that model-level semantic metrics
should be supplemented with targeted instruction-preservation
checks.

## 8. Data-Quality Issues

| sample_id       | data_quality_issue     |   affected_outputs |
|:----------------|:-----------------------|-------------------:|
| general_1510    | source_quality         |                  3 |
| technical_10864 | reference_misalignment |                  3 |
| technical_10967 | reference_misalignment |                  3 |
| technical_11362 | source_context         |                  3 |
| technical_2399  | source_contamination   |                  3 |
| technical_4740  | source_contamination   |                  3 |
| technical_6864  | accelerator_marker     |                  3 |
| technical_7472  | source_contamination   |                  3 |
| technical_7642  | source_contamination   |                  3 |

Nine source sentences, corresponding to 27 model outputs,
require additional verification.

These outputs were excluded from the primary error-rate
calculations.

## 9. Additional Targeted Findings

A separate targeted placeholder evaluation identified the
loss of the %1 placeholder in technical_8912 for E0, E1
and E2.

These targeted examples are not part of the 300-row manual
sample and are therefore not included in the model comparison
denominators.

## 10. Limitations

1. The sample contains deliberately selected high-risk sentences.
2. Only one primary error category is assigned per output.
3. Secondary errors may be documented in reviewer notes.
4. Nine sentences remain subject to data-quality verification.
5. Human annotations are not independently adjudicated.
6. The evaluation does not establish statistical significance.

## 11. Conclusions

The manual evaluation identifies a trade-off between
technical terminology adaptation and semantic reliability.

E1 and E2 show fewer primary terminology errors than E0
but more semantic errors in this sample.

All three models share a critical lost-negation failure.

Future evaluation should prioritize:

- Negation and instruction-preservation tests
- Technical terminology consistency
- Placeholder and filename integrity
- Representative general-domain regression testing
- Resolution of flagged source/reference issues

## 12. Reproducibility

Primary annotation file:

`results/day13_error_analysis/error_annotations.csv`

Generated analysis files:

- `final_model_comparison.csv`
- `final_error_categories.csv`
- `final_severity_comparison.csv`
- `final_domain_comparison.csv`
- `final_comet_comparison.csv`
- `final_data_quality_issues.csv`

All summary statistics are derived from the completed
manual annotation dataset.
