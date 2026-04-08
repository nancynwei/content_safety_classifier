# ---------------------------------------------------------
# Copyright (c) 2026 Nancy N. Wei. All rights reserved.
# No permission is granted to use, copy, modify, distribute, or create
# derivative works without prior written permission from the author.
# See LICENSE at the project root.
# ---------------------------------------------------------
import os
import torch
import json
import numpy as np
import argparse
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import Trainer, DataCollatorWithPadding
from datasets import load_dataset, Value, Sequence
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference on a trained model.")
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True, 
        help="Path to the saved model directory (e.g., ./qwen_finetuned/2026-01-24-final_model)"
    )
    parser.add_argument(
        "--test_file", 
        type=str, 
        required=True, 
        help="Path to the test dataset JSON file (e.g., content_safety_classifier/data/2026-01-24-test_dataset.json)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    saved_model_path = args.model_path
    TEST_FILE = args.test_file

    print(f"Loading model from: {saved_model_path}")
    print(f"Loading test data from: {TEST_FILE}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(saved_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(saved_model_path)
    except OSError as e:
        print(f"\nError loading model: {e}")
        print("Please ensure --model_path points to a valid directory containing config.json and model.safetensors.")
        return

    # Move to GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    try:
        dataset = load_dataset("json", data_files={"test": TEST_FILE})
    except FileNotFoundError:
        print(f"\nError: Test file not found at {TEST_FILE}")
        return

    LABEL_KEYS = ["email_address", "address", "drugs", "bioweapon"]

    def preprocess(example):
        lab = example["labels"]
        if isinstance(lab, dict):
            example["labels"] = [int(lab.get(k, 0)) for k in LABEL_KEYS]
        else:
            example["labels"] = [int(x) for x in lab]
        return example

    dataset = dataset.map(preprocess)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset["test"] = tokenized_dataset["test"].cast_column("labels", Sequence(Value("float32")))

    # Keep 'text' column in original dataset for saving later, but remove from tokenized version for PyTorch
    tokenized_test = tokenized_dataset["test"].remove_columns(["text"])
    tokenized_test.set_format("torch")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs > 0.5).astype(int)
        
        # Flatten the arrays to calculate global (micro) metrics
        labels_flat = labels.flatten()
        preds_flat = preds.flatten()
        
        # Calculate Confusion Matrix
        # labels=[0, 1] ensures we get a 2x2 matrix even if some classes are missing
        tn, fp, fn, tp = confusion_matrix(labels_flat, preds_flat, labels=[0, 1]).ravel()
        
        # Calculate Rates
        # FPR = FP / (FP + TN)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # FNR = FN / (FN + TP)
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

        f1 = f1_score(labels, preds, average="micro")
        acc = accuracy_score(labels, preds)
        
        return {
            "accuracy": acc,
            "f1": f1,
            "false_positive_rate": fpr,
            "false_negative_rate": fnr
        }

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    print("\n" + "="*30)
    print("RUNNING INFERENCE ON FULL TEST SET")
    print("="*30)

    # Generate Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    eval_output_dir = "content_safety_classifier/evaluation"
    os.makedirs(eval_output_dir, exist_ok=True)

    # Run Prediction
    prediction_output = trainer.predict(tokenized_test)

    # Extract Logic
    logits = prediction_output.predictions
    metrics = prediction_output.metrics

    # Convert Logits to Probs
    probs = 1 / (1 + np.exp(-logits))
    THRESHOLD = 0.5
    predictions = (probs > THRESHOLD).astype(int)

    metrics_filename = f"{timestamp}-results.json"
    metrics_file = os.path.join(eval_output_dir, metrics_filename)

    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Metrics saved to {metrics_file}")
    print("Metrics Summary:", metrics)

    inference_data = []

    print(f"Processing {len(dataset['test'])} samples...")

    for i, example in enumerate(dataset["test"]):
        pred_row = predictions[i]
        
        result_entry = {
            "text": example["text"],
            "email_address": int(pred_row[0]),
            "address":       int(pred_row[1]),
            "drugs":         int(pred_row[2]),
            "bioweapon":     int(pred_row[3])
        }
        inference_data.append(result_entry)

    inference_filename = f"{timestamp}-inference_results.json"
    inference_file = os.path.join(eval_output_dir, inference_filename)

    with open(inference_file, "w", encoding="utf-8") as f:
        json.dump(inference_data, f, indent=4)

    print(f"Inference predictions saved to {inference_file}")

if __name__ == "__main__":
    main()