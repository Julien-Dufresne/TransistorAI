# Day 7 — Fine-Tuned Model Evaluation

## Day 5 vs Day 7 Comparison

This report compares the pretrained MarianMT baseline with the fine-tuned model.

Positive deltas indicate improvement; negative deltas indicate degradation.

| Language | Domain | BLEU Baseline | BLEU Fine-Tuned | BLEU Delta | chrF Baseline | chrF Fine-Tuned | chrF Delta |
|---|---|---:|---:|---:|---:|---:|---:|
| de-en | technical | 35.1068 | 40.5126 | +5.4058 | 57.9039 | 61.8439 | +3.9400 |
| de-en | general | 40.4083 | 30.6843 | -9.7240 | 65.4857 | 57.3025 | -8.1832 |

## Interpretation

### de-en — technical

- BLEU improved by 5.4058 points.
- chrF improved by 3.9400 points.

### de-en — general

- BLEU declined by 9.7240 points.
- chrF declined by 8.1832 points.

