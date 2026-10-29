# Day 7 — Fine-Tuned Model Evaluation

**Experiment:** day7_finetuned_de_en
**Model:** results/day8-de-en-v2/technical-finetuned/final_model
**Language pair:** de-en
**Device:** mps

## Evaluation configuration

- Batch size: 8
- Decoding: `{"num_beams": 4, "do_sample": false, "max_new_tokens": 128}`
- Metrics: SacreBLEU BLEU and chrF (Day 5 implementation)

## Evaluation results

| Domain | Sentences | BLEU | chrF | Sentences/sec |
|---|---:|---:|---:|---:|
| technical | 12623 | 40.5126 | 61.8439 | 9.060 |
| general | 1997 | 30.6843 | 57.3025 | 7.329 |

## Reproducibility

- SacreBLEU version: 2.6.0
- BLEU signature: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`
- chrF signature: `nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0`

The evaluation reuses the Day 5 metric implementation, tokenization behavior, decoding parameters and batch size.
