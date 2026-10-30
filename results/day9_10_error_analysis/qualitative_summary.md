# Day 9–10 — Qualitative Error Analysis

## Dataset

- Reviewed translations: 50
- Technical: 25
- General: 25
- Models: E1 (technical-only), E2 (mixed-domain)

## Model preferences

- E1 preferred: 6
- E2 preferred: 11
- Tie: 33

## Severity distribution

| severity   |   E1 |   E2 |
|:-----------|-----:|-----:|
| None       |   15 |   20 |
| Minor      |   12 |   10 |
| Moderate   |   13 |   10 |
| Major      |   10 |   10 |

## Methodology

The analysis uses a manually reviewed sample enriched for
disagreements between E1 and E2.

Error categories are multi-label. Each category is counted
at most once per translation.

Model preferences and error frequencies describe this
selected sample and must not be generalized to the full
test population without representative sampling.

## Generated artifacts

- model_preferences.csv
- severity_distribution.csv
- error_category_distribution.csv
- meaning_preservation.csv
- figures/model_preferences.png
- figures/error_categories.png
- figures/error_severity.png
