# ---------------------------------------------------------
# Copyright (c) 2026 Nancy N. Wei. All rights reserved.
# No permission is granted to use, copy, modify, distribute, or create
# derivative works without prior written permission from the author.
# See LICENSE at the project root.
# ---------------------------------------------------------
import json
import argparse
import os
from collections import Counter
from datetime import datetime

LABEL_KEYS = ["email_address", "address", "drugs", "bioweapon"]

def compute_stats(dataset):
    stats = {
        "total_samples": len(dataset),
        "label_counts": Counter(),
        "pattern_counts": Counter()
    }

    for item in dataset:
        # Create a tuple of 0s and 1s representing the label pattern
        labels = tuple(item["labels"][k] for k in LABEL_KEYS)
        stats["pattern_counts"][labels] += 1

        for k in LABEL_KEYS:
            stats["label_counts"][k] += item["labels"][k]

    return stats

def write_stats_to_file(train, val, test, filename):
    with open(filename, "w") as f:
        for name, ds in [("TRAIN", train), ("VAL", val), ("TEST", test)]:
            stats = compute_stats(ds)

            f.write(f"===== {name} DATASET =====\n")
            f.write(f"Total samples: {stats['total_samples']}\n\n")

            f.write("Label counts:\n")
            for k in LABEL_KEYS:
                f.write(f"  {k}: {stats['label_counts'][k]}\n")

            f.write("\nPattern counts:\n")
            for pattern, count in sorted(stats["pattern_counts"].items()):
                f.write(f"  {pattern}: {count}\n")

            f.write("\n\n")

    print(f"Stats written to {filename}")

def load_json(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find file: {filepath}")
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

if __name__ == "__main__":
    date_str = datetime.now().strftime("%Y-%m-%d")
    default_output = f"content_safety_classifier/data/{date_str}-data_stats.txt"
    parser = argparse.ArgumentParser(description="Compute statistics for train/val/test datasets.")
    parser.add_argument("--train", type=str, required=True, help="Path to the training dataset JSON file")
    parser.add_argument("--val", type=str, required=True, help="Path to the validation dataset JSON file")
    parser.add_argument("--test", type=str, required=True, help="Path to the test dataset JSON file")
    parser.add_argument("--output", type=str, default=default_output, help=f"Output file name (default: {default_output})")
    args = parser.parse_args()
    print(f"Loading datasets...")
    print(f"  Train: {args.train}")
    print(f"  Val:   {args.val}")
    print(f"  Test:  {args.test}")
    train_dataset = load_json(args.train)
    val_dataset = load_json(args.val)
    test_dataset = load_json(args.test)

    write_stats_to_file(train_dataset, val_dataset, test_dataset, filename=args.output)