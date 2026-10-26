#day 6: Creating the main training script, using the functions created in src/training_utils.py and using the configuration from configs/training_config.json

import json
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from src.training_utils import (
    load_translation_dataset,
    tokenize_dataset,
    truncation_report,
    inspect_batch_shapes,
)


def main():
    # STEP 1 — Load the experiment configuration.
    with open("configs/training_config.json") as file:
        config = json.load(file)

    # STEP 2 — Set random seeds for reproducibility.
    set_seed(config["seed"])

    # STEP 3 — Load the pretrained tokenizer and model.
    model_id = config["model_id"]

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    # STEP 4 — Load training and validation data.
    train_dataset = load_translation_dataset(
        config["train_path"],
        config["source_column"],
        config["target_column"],
    )

    validation_dataset = load_translation_dataset(
        config["validation_path"],
        config["source_column"],
        config["target_column"],
    )

    # STEP 5 — Measure truncation before tokenization.
    for split_name, dataset in [
        ("train", train_dataset),
        ("validation", validation_dataset),
    ]:
        for column, max_length in [
            (config["source_column"], config["max_source_length"]),
            (config["target_column"], config["max_target_length"]),
        ]:
            report = truncation_report(
                dataset, tokenizer, column, max_length
            )
            print(f"{split_name}/{column}: {report}")

    # STEP 6 — Tokenize both datasets.
    tokenized_train = tokenize_dataset(
        train_dataset,
        tokenizer,
        config["max_source_length"],
        config["max_target_length"],
        config["source_column"],
        config["target_column"],
    )

    tokenized_validation = tokenize_dataset(
        validation_dataset,
        tokenizer,
        config["max_source_length"],
        config["max_target_length"],
        config["source_column"],
        config["target_column"],
    )

    # STEP 7 — Configure dynamic padding.
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # STEP 8 — Inspect one batch.
    sample_count = min(
        config["per_device_train_batch_size"],
        len(tokenized_train),
    )

    if sample_count == 0:
        raise ValueError("Training dataset is empty.")

    sample_batch = data_collator([
        tokenized_train[i] for i in range(sample_count)
    ])

    for name, tensor in sample_batch.items():
        print(f"{name}: {tuple(tensor.shape)}")

    # STEP 9 — Define training parameters.
    training_args = Seq2SeqTrainingArguments(
        output_dir=config["output_dir"],
        learning_rate=config["learning_rate"],
        optim=config["optimizer"],
        per_device_train_batch_size=config[
            "per_device_train_batch_size"
        ],
        per_device_eval_batch_size=config[
            "per_device_eval_batch_size"
        ],
        gradient_accumulation_steps=config[
            "gradient_accumulation_steps"
        ],
        num_train_epochs=config["num_train_epochs"],
        eval_strategy=config["eval_strategy"],
        save_strategy=config["save_strategy"],
        save_total_limit=config["save_total_limit"],
        seed=config["seed"],
        data_seed=config["seed"],
        predict_with_generate=False,
        report_to="none",
        logging_steps=10,
    )

    # STEP 10 — Initialize the trainer.
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    # STEP 11 — Save the experiment configuration.
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "training_config.json", "w") as file:
        json.dump(config, file, indent=2)

    # STEP 12 — Train and save the final model.
    trainer.train()

    

    # Save the final fine-tuned model.
    trainer.save_model("results/final_model")

    # Save the tokenizer used for training.
    tokenizer.save_pretrained("results/final_model")

    # Save training metrics.
    

    Path("results").mkdir(exist_ok=True)

    with open("results/training_history.json", "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    # STEP 13 — Save training metrics and logs.
    trainer.save_metrics("train", trainer.state.log_history[-1])
    trainer.save_state()

    print("Training completed.")


if __name__ == "__main__":
    main()
