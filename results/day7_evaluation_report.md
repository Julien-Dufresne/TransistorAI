# Day 7 — Fine-Tuned Model Evaluation

## Objective

Evaluate fine-tuning gains on technical translation
and measure general-domain performance changes.

## Experiment Configuration

- Language pair: de-en
- Model: results/final_model
- Seed: 42
- SacreBLEU version: 2.6.0
- Decoding: `{"num_beams": 4, "do_sample": false, "max_new_tokens": 128, "batch_size": 8}`
- Generated (UTC): 2026-10-27T11:04:26.255133+00:00

## Model Comparison

| Domain | BLEU baseline | BLEU fine-tuned | BLEU delta | chrF baseline | chrF fine-tuned | chrF delta |
|---|---:|---:|---:|---:|---:|---:|
| technical | 9.1623 | 26.9278 | +17.7655 | 21.6315 | 54.0667 | +32.4352 |
| general | 3.6273 | 8.7527 | +5.1254 | 24.1778 | 37.3184 | +13.1406 |

## Interpretation

- **Technical**: BLEU improved (+17.7655); chrF improved (+32.4352).
- **General**: BLEU improved (+5.1254); chrF improved (+13.1406).

## Qualitative Error Analysis

Add findings from day7_manual_inspection.csv.

## Limitations

- Automatic metrics do not fully measure adequacy.
- Small score changes may not be statistically significant.
- Dataset and decoding settings must match Day 5.

## Conclusion

As both results improve (both technical and general), fine-tuning may provide broader improvements, not just technical specialization. 