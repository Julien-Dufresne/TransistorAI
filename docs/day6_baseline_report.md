
Day 6 — Reproducible Machine Translation Fine-Tuning Pipeline



1. Implementation

1.1 Dataset loading and preprocessing

Implemented load_translation_dataset() in src/training_utils.py.

The function:

Loads parallel translation pairs from CSV files.

Verifies the source and target columns.

Removes missing or empty translation pairs.

Converts the cleaned data into Hugging Face Dataset objects.

1.2 Tokenization

Implemented tokenize_dataset() to prepare German source sentences and English reference translations.

Key concepts:

input_ids: numerical representations of source tokens.

attention_mask: identifies valid input tokens and padding.

labels: target token IDs used to calculate training loss.

text_target: tokenizer argument used to process target translations.

Source and target sequences are tokenized separately, with truncation enabled.

1.3 Truncation inspection

Implemented truncation_report() to measure the number and proportion of sequences exceeding the configured maximum token length.

This is particularly important for technical translation, where truncation may remove terminology or essential sentence information.

1.4 Training pipeline

Implemented the training workflow in src/train.py using:

AutoTokenizer

AutoModelForSeq2SeqLM

DataCollatorForSeq2Seq

Seq2SeqTrainingArguments

Seq2SeqTrainer

Dynamic padding was used to prepare variable-length translation examples for training.

2. Experiment configuration

Hyperparameter/          Value

Model/                   Helsinki-NLP/opus-mt-de-en (verify actual model ID)
Optimizer/               AdamW
Learning rate/           2e-5 initial setting
Random seed/             42
Batch size per device/   4
Gradient accumulation/   2
Effective batch size/    8 on one device
Epochs/                  3
Maximum source length/   128
Maximum target length/   128
Evaluation strategy/     Every epoch
Checkpoint strategy/     Every epoch
Hardware/                Apple Silicon / PyTorch MPS

The training configuration is stored in configs/training_config.json to support reproducibility.

The exact installed library versions, dataset revision and pretrained model revision should also be recorded for future experiments.

3. Training results

The fine-tuning experiment completed successfully after three epochs.

3.1 Execution summary

Metric/                         Result
Training duration/              5 h 37 min 55 sec
Total optimizer steps/          36,423
Completed epochs/               3
Final overall training loss/    2.994
Final validation loss/          1.175
Training status/                Completed

3.2 Validation loss progression

Epoch/      Validation loss/        Relative change

1/          1.369/                  —

2/          1.216/                  −11.2%

3/          1.175/                  −3.4%

Total validation loss reduction: approximately 14.2% between epochs 1 and 3.

3.3 Interpretation

Validation loss decreased consistently across all three epochs, indicating that the model became more effective at predicting reference translation tokens on the validation dataset.

No increase in validation loss was observed during the experiment.

However, lower validation loss does not necessarily guarantee improved translation quality. BLEU and chrF evaluation on the held-out test set is required before concluding that fine-tuning outperforms the pretrained baseline.

4. Technical challenges and solutions

4.1 Undefined variables in training utilities (resolved)

Issue: Pylance reported undefined variables for tokenizer, model, and tokenized_train.

Cause: Batch inspection code referenced variables that were not defined in its scope.

Resolution: Removed the unnecessary standalone inspection code from training_utils.py, retaining the reusable dataset loading, tokenization and truncation functions.

Learning: Utility functions should receive dependencies explicitly through parameters rather than relying on variables defined elsewhere.

5.2 MPS pinned-memory warning

Issue: PyTorch reported that pinned memory was unsupported on the MPS backend.

Resolution: Identified this as a non-fatal warning. Training continued successfully.

Learning: Hardware-specific warnings should be distinguished from errors that affect model correctness.

5.3 Long training duration

Challenge: Full fine-tuning required approximately 5 hours and 38 minutes.

Learning: Training runtime depends on dataset size, batch configuration, sequence lengths, hardware acceleration and evaluation overhead.

Short smoke tests can validate the pipeline before committing to a full training run.

6. Key learnings

Supervised sequence-to-sequence learning: The model learns to predict target-language tokens conditioned on source-language input.

Tokenization: Text must be converted into token IDs before being processed by the Transformer.

Dynamic padding: Batches can contain sequences of different lengths while minimizing unnecessary padding.

Gradient accumulation: Multiple smaller batches can contribute to one optimizer update.

Learning-rate scheduling: The learning rate can decrease over training to support progressively smaller updates.

Validation loss: Held-out data helps monitor model generalization and detect potential overfitting.

Reproducibility: Configuration files, random seeds, dataset versions and model checkpoints are essential for repeatable experiments.

Experiment monitoring: Training logs help track loss, gradient norms, learning rates, epoch progress and runtime.

7. Deliverables

src/training_utils.py — Dataset preparation and tokenization utilities.

src/train.py — Reproducible fine-tuning pipeline.

configs/training_config.json — Experiment hyperparameters.

results/training_history.json — Training history.

results/final_model/ — Fine-tuned model artifacts, stored locally.

docs/day6_training_report.md — Experiment documentation.

8. Limitations

The experiment used one initial hyperparameter configuration.

No systematic hyperparameter search was performed.

Translation quality has not yet been assessed using BLEU or chrF after fine-tuning.

Validation loss alone cannot establish domain-specific translation improvements.

Reproducibility may vary across library versions and hardware backends.

9. Next steps — Day 7

Load the saved fine-tuned translation model.

Generate predictions on the held-out technical and general-domain test datasets.

Calculate SacreBLEU BLEU and chrF scores.

Compare the fine-tuned model against the Day 5 pretrained baseline.

Analyze whether fine-tuning improves technical translation while preserving general-domain performance.

Save a reproducible evaluation report.

Conclusion

Day 6 successfully established a reproducible machine translation fine-tuning workflow using PyTorch and Hugging Face.

The model completed three epochs and achieved a validation loss reduction of approximately 14.2% between epochs 1 and 3.

The next stage is to determine whether these improvements translate into measurable gains in translation quality using standardized evaluation metrics.
First test with max_step 5:
[
  {
    "eval_loss": 4.5567402839660645,
    "eval_runtime": 111.4741,
    "eval_samples_per_second": 109.021,
    "eval_steps_per_second": 27.262,
    "epoch": 0.0004118276912939626,
    "step": 5
  },
  {
    "train_runtime": 114.9925,
    "train_samples_per_second": 0.348,
    "train_steps_per_second": 0.043,
    "total_flos": 813560758272.0,
    "train_loss": 9.559857940673828,
    "epoch": 0.0004118276912939626,
    "step": 5
  }
]