# ---------------------------------------------------------
# Copyright (c) 2026 Nancy N. Wei. All rights reserved.
# No permission is granted to use, copy, modify, distribute, or create
# derivative works without prior written permission from the author.
# See LICENSE at the project root.
# ---------------------------------------------------------
import argparse
import json
import os
import shutil
import torch
from datetime import datetime
from datasets import load_dataset, Value, Sequence
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from sklearn.metrics import f1_score, accuracy_score

def train_model(train_file, val_file, test_file, output_dir):
    print(f"Loading data...")
    print(f"  Train: {train_file}")
    print(f"  Val:   {val_file}")
    print(f"  Test:  {test_file}")

    dataset = load_dataset(
        "json",
        data_files={"train": train_file, "validation": val_file, "test": test_file}
    )

    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    LABEL_KEYS = ["email_address", "address", "drugs", "bioweapon"]


    def encode_labels(example):
        lab = example["labels"]
        if isinstance(lab, dict):
            example["labels"] = [int(lab.get(k, 0)) for k in LABEL_KEYS]
        else:
            example["labels"] = [int(x) for x in lab]
        return example

    dataset = dataset.map(encode_labels)


    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    def tokenize_function(examples):
        return tokenizer(
            examples["text"], truncation=True, padding="max_length", max_length=512
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Fix Float32 Issue for PyTorch
    for split in tokenized_dataset.keys():
        tokenized_dataset[split] = tokenized_dataset[split].cast_column(
            "labels", Sequence(Value("float32"))
        )
    
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset.set_format("torch")


    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=4, problem_type="multi_label_classification"
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    temp_dir = "./temp_checkpoints"
    
    training_args = TrainingArguments(
        output_dir=temp_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=5e-5,
        num_train_epochs=3,
        eval_strategy="steps",
        save_strategy="steps",
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits = torch.tensor(eval_pred.predictions)
        labels = torch.tensor(eval_pred.label_ids)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).int().cpu().numpy()
        labels = labels.int().cpu().numpy()
        return {
            "f1": f1_score(labels, preds, average="micro"),
            "accuracy": accuracy_score(labels, preds)
        }

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    # Clean up temp folder
    try:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("Freed up space by deleting intermediate checkpoints.")
    except Exception as e:
        print(f"Warning: Could not delete temp checkpoints: {e}")

    # Save Final Model
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving final model to {output_dir}...")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully!")


    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Path to training data")
    parser.add_argument("--val_file", type=str, required=True, help="Path to validation data")
    parser.add_argument("--test_file", type=str, required=True, help="Path to test data")
    
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    default_output = f"./content_safety_classifier/model/checkpoint/{date_str}-model"
    parser.add_argument("--output_dir", type=str, default=default_output)

    args = parser.parse_args()

    train_model(args.train_file, args.val_file, args.test_file, args.output_dir)