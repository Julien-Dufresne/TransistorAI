# TransistorAI — Reproducibility Guide

## Environment

Development environment: Apple Silicon macOS.

- Python: 3.11.18
- PyTorch: 2.15.0
- Transformers: 4.44.2
- Datasets: 3.6.0
- Streamlit: 1.39.0
- MLX: 0.32.4
- MLX-LM: 0.32.2
- COMET: 2.2.7
- SacreBLEU: 2.6.0

The complete environment snapshot is stored in `requirements-lock.txt`.

## Translation models

| Model | Description | Location |
|---|---|---|
| E0 | Pretrained MarianMT | Helsinki-NLP/opus-mt-de-en |
| E1 | Technical fine-tuned MarianMT | results/day8-de-en-v2/technical-finetuned/final_model/ |
| E2 | Mixed fine-tuned MarianMT | results/day8-de-en-v2/mixed-finetuned/final_model/ |
| E3 | Quantized Qwen2.5 LLM | mlx-community/Qwen2.5-3B-Instruct-4bit |

E1 and E2 weights are stored locally and are not tracked by Git.
E0 and E3 require their model files to be cached or downloaded.

## Datasets

| Domain | Train | Validation | Test |
|---|---:|---:|---:|
| Technical | 101,401 | 12,662 | 12,623 |
| General | 25,001 | 2,499 | 1,997 |

Frozen Day 8 datasets: `data/versions/day8-de-en-v2/`

Working datasets: `data/processed/`

These datasets are not fully tracked by Git and must be restored
or reconstructed for independent reproduction.

## Training parameters

Recorded E1 configuration:

- Learning rate: 2e-5
- Batch size: 4
- Gradient accumulation: 2
- Epochs: 3
- Maximum sequence length: 128
- Seed: 42

Consult experiment-specific training configurations for E2.

## Installation

Use Python 3.11 on Apple Silicon macOS.

Install the recorded packages:

    python -m pip install -r requirements-lock.txt

The lockfile records the working environment but does not
guarantee cross-platform compatibility.

## Validation

Check dataset integrity:

    python src/check_e2_datasets.py

Run the translation smoke tests:

    python evaluation/day18_translation_smoke.py

The latest reported smoke-test result was 20/20 PASS.

Smoke tests check execution, not translation accuracy.

## Streamlit MVP

Start the application:

    python -m streamlit run MVP_streamlit_app.py

The MVP includes translation, practice activities,
human preference collection and evaluation analytics.

## Reproducibility limitations

A fresh Git clone does not include all frozen datasets,
E1/E2 model weights or cached E0/E3 model artifacts.

The COMET evaluation model may also require a download.

Full reproduction requires access to those artifacts,
a compatible environment and the relevant training scripts.
