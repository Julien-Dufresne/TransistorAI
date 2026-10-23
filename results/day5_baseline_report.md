# TransistorAI — Day 5 Baseline Benchmark

## Research objective

Evaluate pretrained MarianMT translation models
before domain-specific fine-tuning.

The objective is to establish reproducible baseline
translation quality across technical and general domains.

## Experimental results

| language_pair   | domain    |   bleu |   chrf |   sentences_per_second |
|:----------------|:----------|-------:|-------:|-----------------------:|
| de-en           | general   |   3.63 |  24.18 |                   6.97 |
| de-en           | technical |   9.16 |  21.63 |                   7.57 |

## Evaluation methodology

- Corpus-level SacreBLEU BLEU and chrF
- Held-out technical and general-domain test sets
- Fixed decoding configuration
- One reference translation per source sentence
- Batched inference using PyTorch

## Metric and model configuration

```json
[
  {
    "language_pair": "de-en",
    "domain": "general",
    "model_id": "Helsinki-NLP/opus-mt-de-en",
    "decoding": {
      "num_beams": 4,
      "do_sample": false,
      "max_new_tokens": 128
    },
    "device": "mps",
    "batch_size": 8,
    "dataset": "data/processed/general/de-en/general_test.csv",
    "bleu_signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0",
    "chrf_signature": "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0",
    "sacrebleu_version": "2.6.0"
  },
  {
    "language_pair": "de-en",
    "domain": "technical",
    "model_id": "Helsinki-NLP/opus-mt-de-en",
    "decoding": {
      "num_beams": 4,
      "do_sample": false,
      "max_new_tokens": 128
    },
    "device": "mps",
    "batch_size": 8,
    "dataset": "data/processed/technical/de-en/technical_test.csv",
    "bleu_signature": "nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0",
    "chrf_signature": "nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0",
    "sacrebleu_version": "2.6.0"
  }
]
```

## Interpretation

Compare technical and general-domain performance
within each language direction.

Higher BLEU and chrF scores generally indicate
greater agreement with reference translations.

Differences between domains may reflect differences
in vocabulary, sentence complexity and test data.

Avoid directly comparing BLEU scores across language
directions as though they represent identical tasks.

## Limitations

BLEU and chrF do not fully measure:

- Semantic accuracy
- Technical terminology correctness
- Hallucinations
- Omissions
- Contextual appropriateness

## Next steps

1. Inspect low-quality translations manually.
2. Analyze terminology and semantic errors.
3. Prepare domain-specific fine-tuning experiments.
4. Compare fine-tuned models against this baseline.
