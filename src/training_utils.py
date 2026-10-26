# Day 6
# This module contains reusable dataset loading and tokenization functions for translation tasks.

import pandas as pd
from datasets import Dataset


def load_translation_dataset(
    csv_path,
    source_column="source",
    target_column="target",
):
    """Load a parallel translation dataset from CSV."""

    df = pd.read_csv(csv_path)

    # Verify the expected columns exist.
    required = [source_column, target_column]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Remove incomplete translation pairs.
    df = df.dropna(subset=required).copy()

    # Convert text to strings and remove empty pairs.
    for col in required:
        df[col] = df[col].astype(str).str.strip()

    df = df[
        (df[source_column] != "")
        & (df[target_column] != "")
    ]

    return Dataset.from_pandas(
        df[required],
        preserve_index=False,
    )


def tokenize_dataset(
    dataset,
    tokenizer,
    max_source_length=128,
    max_target_length=128,
    source_column="source",
    target_column="target",
):
    """Tokenize source sentences and target translations."""

    def preprocess(examples):

        # 1. Tokenize the source sentences (e.g. German).
        model_inputs = tokenizer(
            examples[source_column],
            max_length=max_source_length,
            truncation=True,
        )

        # 2. Tokenize the target translations (e.g. English).
        labels = tokenizer(
            text_target=examples[target_column],
            max_length=max_target_length,
            truncation=True,
        )

        # 3. Add target token IDs as training labels.
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    # 4. Apply tokenization to the entire dataset.
    return dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names,
    )

# Inspect truncation : if a sentence exceeds the maximum token length, it will be truncated. This function checks how many sentences are affected by truncation.
def truncation_report(dataset, tokenizer, column, max_length):
    """Check how many sentences exceed the maximum token length."""

    # Calculate token lengths without truncation.
    lengths = [
        len(tokenizer(str(text), truncation=False)["input_ids"])
        for text in dataset[column]
    ]

    # Count sentences exceeding the maximum length.
    truncated = sum(length > max_length for length in lengths)

    return {
        "total_sentences": len(lengths),
        "truncated_sentences": truncated,
        "truncation_rate": truncated / len(lengths) if lengths else 0,
        "maximum_length": max(lengths, default=0),
    }

# Inspect batch shapes.
# This function checks whether tokenized examples
# can be correctly combined into training batches.

from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader


def inspect_batch_shapes(
    tokenized_dataset,
    tokenizer,
    model,
    batch_size=4,
):
    """Inspect the shapes of a tokenized training batch."""

    # 1. Create a data collator.
    # It dynamically pads source and target sequences.
    collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # 2. Create a DataLoader.
    # It groups examples into batches.
    loader = DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        collate_fn=collator,
    )

    # 3. Retrieve the first batch.
    batch = next(iter(loader))

    # 4. Display the shape of each tensor.
    for key, value in batch.items():
        print(f"{key}: {value.shape}")

    return batch