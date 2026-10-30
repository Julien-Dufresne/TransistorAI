# TransistorAI 

German-to-English technical-domain machine translation with MarianMT, a Streamlit demo, reproducible dataset preparation, supervised fine-tuning, and baseline comparison.

## What changed

- Rebuilt the project from selected source and configuration files, separating generated datasets and model weights from versioned code.
- Standardized German source and English target columns, including the general-domain preparation script.
- Corrected the manual inspection script's project-root lookup.
- Made the baseline select the two held-out test sets explicitly through `configs/project_config.json`; baseline reports use `results/`.
- Grouped technical splits by German source sentence with seed 42 and added quality filtering and overlap validation.
- Saved the fine-tuned model, configuration, training history, and aggregate metrics under the configured run directory.
- Evaluated the fine-tuned model with the baseline's decoding, batch size, and metric implementation, and recorded both domain improvements and regressions.

## Results

| Domain | Test pairs | Baseline BLEU | Fine-tuned BLEU | BLEU change | Baseline chrF | Fine-tuned chrF | chrF change |
|---|---:|---:|---:|---:|---:|---:|---:|
| Technical | 12,623 | 35.1068 | 40.5126 | +5.4058 | 57.9039 | 61.8439 | +3.9400 |
| General | 1,997 | 40.4083 | 30.6843 | -9.7240 | 65.4857 | 57.3025 | -8.1832 |

Technical-domain fine-tuning improved the technical test scores but reduced general-domain quality. These are single-run results; statistical significance and human preference have not been established. Manual inspection files are generated locally; creating samples alone does not establish that a human review is complete.

Training used 101,401 technical pairs and 12,662 validation pairs, three epochs, learning rate 0.00002, batch size 4, gradient accumulation 2, maximum source/target length 128, and seed 42. Recorded runtime was about three hours. The exported model is from the final epoch, rather than an explicitly selected best checkpoint.

The `day8-de-en-v2/technical-finetuned` directory name is retained to match the completed run and saved model location; this branch documents the Days 1–7 workflow.

## Layout

```text
configs/              Project, decoding, metrics, training, evaluation settings
src/                  Dataset preparation, baseline, training, app helpers
evaluation/           Metrics, validators, comparison, manual sample scripts
data/README.md        Dataset handling instructions
results/              Aggregate metrics, reports, and training logs
MVP_streamlit_app.py  Translation demo
baseline_translation.py  Initial translation and latency experiment
```

Raw/processed corpora, sentence-level predictions and inspection samples, checkpoints, and exported weights are excluded from this branch's snapshot. Earlier Git history may still contain datasets from the original project.

## Recreate Days 1–7

Run commands from the repository root.

### Day 1 — Environment

```bash
conda create -n transistorai-fresh python=3.11
conda activate transistorai-fresh
python -m pip install -r requirements.txt
```

Requirements are unpinned; exact historical dependency versions are not recoverable from this file alone. Record a verified environment with `python -m pip freeze > requirements-lock.txt`.

### Day 2 — Pretrained translation

```bash
python baseline_translation.py
```

The configured direction is German to English. Some legacy console labels in this initial script still use reversed EN/DE labels.

### Day 3 — Streamlit demo

```bash
python -m streamlit run MVP_streamlit_app.py
```

### Day 4 — Data preparation

Obtain aligned OPUS KDE4 v2 and WMT19 newstest2019 text files, observing their source licenses and citation requirements. Place files at:

```text
data/raw/technical/de-en/KDE4.de-en.de
data/raw/technical/de-en/KDE4.de-en.en
data/raw/general/de-en/newstest2019.de
data/raw/general/de-en/newstest2019.en
```

Record source versions, download provenance, licenses, and checksums locally before preparing data. The preparation scripts do not download these corpora.

```bash
python -m src.prepare_dataset
python -m src.prepare_general_test
python -m evaluation.validate_datasets
python -m evaluation.dataset_statistics
python evaluation/manual/manual_dataset_inspection.py
```

Review `evaluation/manual_dataset_inspection.csv`. Validation checks nulls, empty text, exact pair duplicates, and exact source overlap; it does not establish translation correctness or absence of near-duplicates. Keep test splits fixed and out of training and checkpoint selection.

### Day 5 — Baseline

```bash
python -m src.benchmark && python -m evaluation.generate_baseline_report
```

### Day 6 — Fine-tuning

```bash
python -m src.train
```

The current configuration exports the model to `results/day8-de-en-v2/technical-finetuned/final_model` and checkpoints to `models/checkpoints/de-en`.

### Day 7 — Evaluation

```bash
python -m evaluation.evaluate_finetuned && python -m evaluation.compare_models
python -m evaluation.manual_day7_inspection
```

Review `results/day7_comparison.csv`, both Day 7 reports, and the locally generated manual inspection sample. Rerunning report commands overwrites generated reports.
