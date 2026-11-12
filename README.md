# TransistorAI

**German-to-English Machine Translation — Domain Adaptation, LLM Comparison and Human Evaluation**

TransistorAI is an applied machine learning project comparing pretrained neural machine translation (NMT), domain-adapted NMT, and a quantized large language model (LLM).

The project investigates how fine-tuning affects translation quality across technical and general-language domains, and how automatic evaluation metrics relate to human preferences.

## Research Objectives

1. Establish a pretrained translation baseline.
2. Fine-tune MarianMT on technical-domain data.
3. Compare technical-only and mixed-domain fine-tuning.
4. Evaluate an LLM translation baseline.
5. Compare BLEU, chrF and COMET with human preferences.
6. Deliver an interactive Streamlit MVP.

## Translation Systems

| ID | System | Description |
|---|---|---|
| E0 | MarianMT baseline | Pretrained `Helsinki-NLP/opus-mt-de-en` |
| E1 | Technical fine-tuned | MarianMT trained on technical-domain data |
| E2 | Mixed fine-tuned | MarianMT trained on mixed-domain data |
| E3 | LLM baseline | `mlx-community/Qwen2.5-3B-Instruct-4bit` |

E1 and E2 are stored as local model exports. E3 runs through MLX on Apple Silicon.

## Datasets

| Domain | Training | Validation | Test |
|---|---:|---:|---:|
| Technical | 101,401 | 12,662 | 12,623 |
| General | 25,001 | 2,499 | 1,997 |

The frozen Day 8 dataset snapshot is stored under:

`data/versions/day8-de-en-v2/`

Working datasets are stored under:

`data/processed/`

These artifacts are not fully tracked by Git. See [Reproducibility Guide](docs/reproducibility.md) for details.

## Experimental Results

The following results were obtained on the full technical and general test sets.

### Technical Domain

| Model | BLEU ↑ | chrF ↑ | COMET ↑ |
|---|---:|---:|---:|
| E0 | 35.1068 | 57.9039 | 0.8378 |
| **E1** | **40.5126** | **61.8439** | **0.8676** |
| E2 | 39.8935 | 61.4829 | 0.8660 |
| E3 | 29.1308 | 54.0144 | 0.8113 |

E1 achieved the strongest results on the technical test set.

### General Domain

| Model | BLEU ↑ | chrF ↑ | COMET ↑ |
|---|---:|---:|---:|
| **E0** | **40.4083** | **65.4857** | **0.8668** |
| E1 | 30.6843 | 57.3025 | 0.8266 |
| E2 | 35.6534 | 61.6582 | 0.8493 |
| E3 | 33.5082 | 61.1845 | 0.8584 |

E0 achieved the strongest results on the general test set.

### Main Findings

- Technical-only fine-tuning improved technical translation quality.
- Technical-only fine-tuning reduced general-domain performance.
- Mixed-domain fine-tuning provided a compromise between specialization and general-language retention.
- The quantized LLM baseline did not outperform the pretrained NMT baseline on these full test sets.
- Translation quality depends on domain, training data and evaluation methodology.

## Human Preference Evaluation

A human evaluation interface was developed to compare translations from E0, E1, E2 and E3.

The evaluation dataset contains 200 German sentences:

- 100 technical sentences
- 100 general-domain sentences

Reviewers can compare alternative translations and select their preferred output.

Selections are stored in SQLite for subsequent analysis.

### Pilot Results

The Day 17 pilot contained:

- 25 human preference votes
- 21 unique evaluated sentences

Strict agreement between human selections and automatic metric rankings:

| Metric | Agreement |
|---|---:|
| BLEU | 36% |
| chrF | 28% |
| COMET | 36% |

These are preliminary pilot results. The sample size is insufficient for broad conclusions about metric reliability.

## Streamlit MVP

The interactive application includes:

- German-to-English translation
- Model selection
- Translation practice activities
- Vocabulary learning
- Human preference collection
- Human evaluation analytics
- Project exploration and evaluation results

### Launch

```bash
python -m streamlit run MVP_streamlit_app.py
```

The application normally opens at `http://localhost:8501`.

Local model files and cached model dependencies must be available for the relevant translation systems.

## Installation

The verified development environment uses Python 3.11.18 on Apple Silicon macOS.

Install dependencies using:

```bash
python -m pip install -r requirements-lock.txt
```

Alternatively, `requirements.txt` contains the project's general dependency requirements.

The lockfile records the local development environment and may require adaptation on other hardware or operating systems.

For dataset restoration, model locations, environment details and reproducibility limitations, see:

**[docs/reproducibility.md](docs/reproducibility.md)**

## Validation and Testing

Check dataset integrity:

```bash
python src/check_e2_datasets.py
```

Run the translation smoke tests:

```bash
python evaluation/day18_translation_smoke.py
```

The latest reported Day 18 smoke-test execution passed **20/20 checks**.

Smoke tests verify execution behavior; they do not establish semantic translation accuracy.

Translation quality was evaluated separately using BLEU, chrF, COMET, terminology checks, qualitative error analysis and human preferences.

## Repository Structure

```text
TransistorAI-fresh/
├── MVP_streamlit_app.py
├── data/
│   ├── processed/
│   ├── versions/
│   └── human_evaluation/
├── docs/
│   └── reproducibility.md
├── evaluation/
├── results/
├── src/
├── requirements.txt
├── requirements-lock.txt
└── README.md
```

The repository also contains experiment reports, evaluation scripts and additional Streamlit pages.

## Limitations

- Fine-tuned model weights and frozen datasets are not fully distributed through Git.
- Some evaluation workflows require downloaded or cached models.
- MLX-based LLM inference is hardware-dependent.
- Human preference results currently represent a small pilot.
- BLEU, chrF and COMET cannot capture every aspect of translation quality.
- Exact-pair dataset overlap checks do not rule out all possible data leakage.

## Conclusion

TransistorAI demonstrates that domain adaptation can improve specialized translation quality while introducing trade-offs in general-domain performance.

The project combines neural machine translation, an LLM baseline, automatic metrics, qualitative analysis, human evaluation and an interactive MVP.

The results support a domain-aware approach to translation model selection rather than assuming that one model is optimal for every use case.

## Historical Documentation

The original Days 1–7 setup instructions are preserved in:

[docs/README_days1-7_legacy.md](docs/README_days1-7_legacy.md)


python -m streamlit run MVP_streamlit_app.py 
or 
fn + F5 to launch the app, Shift + fn +F5 to close the app (or CTRL + C)