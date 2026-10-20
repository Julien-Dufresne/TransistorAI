# Project details

A five-week, research-first project on English-to-German neural machine translation, technical-domain adaptation, and quality evaluation. The project starts with a small Streamlit translation MVP, then compares a pretrained NMT baseline, supervised fine-tuned models, and a multilingual LLM.

## Research question

Can supervised fine-tuning improve English-to-German translation of technical text, and how does a domain-adapted NMT model compare with a multilingual LLM in translation quality, terminology accuracy, generalization, and inference efficiency?

## Project goals

- Establish reproducible pretrained NMT baselines before fine-tuning.
- Compare technical and general-domain performance while keeping held-out test sets isolated.
- Run controlled supervised fine-tuning experiments and document positive, neutral, or negative findings.
- Evaluate with BLEU, chrF, and COMET, supported by terminology checks, error analysis, and latency measurements.
- Compare baseline NMT, fine-tuned NMT, and a multilingual LLM; collect a small human preference set.
- Present the findings in an interactive Streamlit research lab.