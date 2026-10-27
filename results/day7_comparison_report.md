# Day 7 — Fine-Tuned Model Evaluation

## Day 5 vs Day 7 Comparison

This report compares the pretrained MarianMT baseline with the fine-tuned model.

Positive deltas indicate improvement; negative deltas indicate degradation.

| Language | Domain | BLEU Baseline | BLEU Fine-Tuned | BLEU Delta | chrF Baseline | chrF Fine-Tuned | chrF Delta |
|---|---|---:|---:|---:|---:|---:|---:|
| de-en | technical | 9.1623 | 26.9278 | +17.7655 | 21.6315 | 54.0667 | +32.4352 |
| de-en | general | 3.6273 | 8.7527 | +5.1254 | 24.1778 | 37.3184 | +13.1406 |

## Interpretation

### de-en — technical

- BLEU improved by 17.7655 points.
- chrF improved by 32.4352 points.

### de-en — general

- BLEU improved by 5.1254 points.
- chrF improved by 13.1406 points.

