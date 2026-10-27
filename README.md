# TransistorAI
End of Study project


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

## Five-week roadmap

| Week | Focus | Milestone |
|---|---|---|
| 1 | Data, pretrained baseline, first MVP | Working EN→DE translator and baseline report |
| 2 | PyTorch fine-tuning and domain adaptation | Fine-tuned model and controlled E0/E1/E2 runs |
| 3 | Evaluation, error analysis, regression checks | BLEU / chrF / COMET pipeline and evaluation report |
| 4 | LLM comparison and preference evaluation | Comparative results and preference analysis |
| 5 | Streamlit research demo and portfolio handoff | Reproducible repository, report, and presentation |

See [`docs/project-checklist.docx`](docs/project-checklist.docx) for the editable day-by-day plan, acceptance criteria, experiment logs, and final submission checklist.

## Initial scope

- **Primary direction:** English → German
- **Secondary direction:** German → English, if time allows
- **Suggested baseline:** `Helsinki-NLP/opus-mt-en-de`
- **Suggested domain:** Software documentation
- **Suggested targets:** 10,000 training pairs, 1,000 validation pairs, 1,000 technical test pairs, and 500 general test pairs. These are targets; prioritize licensed, well-aligned data.

Keep test sets out of training and checkpoint selection. Record dataset source, license, cleaning, split method, model identifiers, metric versions, seeds, and compute context. Report negative results and limitations clearly.

## Repository layout

```text
app/                 Streamlit research interface
configs/             Model and experiment configurations
data/                Dataset instructions only; datasets are not committed
  README.md
notebooks/           Exploratory research notebooks
reports/             Evaluation reports, plots, and presentation material
src/translatelab/    Reusable training, inference, and evaluation code
  __init__.py
```

## Setup

Python environment and dependency versions will be recorded in `requirements.txt` as the implementation begins. Dataset and model access may require separate setup; follow the applicable licenses and terms.

## Optional or future work

FastAPI, Docker, French support, personalized-learning features, MLflow, and full RLHF / GRPO training are outside the core five-week scope. Prioritize fine-tuning, controlled evaluation, and the Streamlit demonstration first.

## Apple role alignment

This project is designed to demonstrate work relevant to the Machine Learning Research Engineer, Machine Translation role and the AIML Applied Research Engineer, Machine Translation role: NMT and Transformer research, supervised fine-tuning, data quality, controlled experiments, evaluation, human preference signals, quality regression checks, and a reliable demonstration.

## Status

Project scaffold created. Implementation work begins with Week 1, Day 1 in the checklist.

## Launch the app:
python -m streamlit run streamlit_app.py

conda activate transistorai-env