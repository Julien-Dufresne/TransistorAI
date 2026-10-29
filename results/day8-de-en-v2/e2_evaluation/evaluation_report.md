# Day 7 — Fine-Tuned Model Evaluation

**Experiment:** day8_e2_mixed_de_en
**Model:** results/day8-de-en-v2/mixed-finetuned/final_model
**Language pair:** de-en
**Device:** mps

## Evaluation configuration

- Batch size: 8
- Decoding: `{"num_beams": 4, "do_sample": false, "max_new_tokens": 128}`
- Metrics: SacreBLEU BLEU and chrF (Day 5 implementation)

## Evaluation results

| Domain | Sentences | BLEU | chrF | Sentences/sec |
|---|---:|---:|---:|---:|
| technical | 12623 | 39.8935 | 61.4829 | 8.580 |
| general | 1997 | 35.6534 | 61.6582 | 7.522 |

## Reproducibility

- SacreBLEU version: 2.6.0
- BLEU signature: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`
- chrF signature: `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`

The evaluation reuses the Day 5 metric implementation, tokenization behavior, decoding parameters and batch size.
