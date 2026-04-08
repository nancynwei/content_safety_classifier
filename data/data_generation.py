# ---------------------------------------------------------
# Copyright (c) 2026 Nancy N. Wei. All rights reserved.
# No permission is granted to use, copy, modify, distribute, or create
# derivative works without prior written permission from the author.
# See LICENSE at the project root.
# ---------------------------------------------------------
import os
import argparse
from xai_sdk import Client
from tqdm import tqdm
import json
import random
from xai_sdk.chat import user, system
from datetime import datetime


# Function to read API Key from a file
def load_api_key(filename="grok_key.txt"):
    try:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                return f.read().strip()
        elif os.path.exists(os.path.join("..", filename)):
             with open(os.path.join("..", filename), "r") as f:
                return f.read().strip()
        else:
            print(f"Warning: '{filename}' not found. Using default/environment key.")
            return os.getenv("GROK_API_KEY", "HIDDEN")
    except Exception as e:
        print(f"Error reading key file: {e}")
        return "HIDDEN"

    
# Load key from file and create client
GROK_API_KEY = load_api_key("grok_key.txt")
client = Client(api_key=GROK_API_KEY, timeout=3600)


def generate_data(train_samples=250, val_samples=75, test_samples=75):
    """
    Generates training, validation, and test datasets.
    """ 
    # -----------------------------
    # Label specs & Constants
    # -----------------------------
    CONTENT_TYPES = [
        "news article", "press release", "advertisement", 
        "marketing email", "social media post", "online forum comment"
    ]

    SYSTEM_PROMPT = (
        "You are a content safety assistant generating realistic online text. "
        "The text may resemble news articles, press releases, advertisements, "
        "emails, social media posts, or public forum comments. "
        "The writing should feel natural, human, and contextually realistic. "
        "Use multiple sentences and appropriate tone for the format. "
        "Some texts may be informal, others formal. "
        "Not all text should include emojis. "
        "Do NOT explain labels or intentions. "
        "Output ONLY the text."
    )

    # -----------------------------
    # User prompt builder
    # -----------------------------
    def build_user_prompt(labels, ambiguous_rate=0.3):
        email, address, drugs, bioweapon = labels
        ambiguous = random.random() < ambiguous_rate
        content_type = random.choice(CONTENT_TYPES)
        constraints = []

        if email:
            constraints.append("Include something that looks like an email address to a human, possibly malformed.")
        if address:
            constraints.append("Include something that resembles a physical address or location.")
        if drugs:
            if ambiguous:
                constraints.append("Include a vague or slang reference that could plausibly refer to drugs.")
            else:
                constraints.append("Include a casual reference to illegal drugs or drug culture.")
        if bioweapon:
            if ambiguous:
                constraints.append("Include a high-level or metaphorical reference to biological danger.")
            else:
                constraints.append("Include a non-technical mention of dangerous biological agents.")
        if not constraints:
            constraints.append("Contain no personal data, drugs, or weapons of any kind.")

        constraint_text = " ".join(constraints)
        return (
            f"Write a realistic {content_type}. Use 3–6 sentences. "
            f"{constraint_text} Do not explain anything. Output only the text."
        )

    # -----------------------------
    # Dataset generator
    # -----------------------------
    def generate_synthetic_dataset(client, label_patterns, samples_per_pattern, save_every=10, save_path="dataset.json"):
        dataset = []
        total_samples = len(label_patterns) * samples_per_pattern
        counter = 0

        with tqdm(total=total_samples, desc=f"Generating {save_path}") as pbar:
            for pattern_idx, labels in enumerate(label_patterns, start=1):
                for sample_idx in range(1, samples_per_pattern + 1):
                    try:
                        chat = client.chat.create(
                            model="grok-4-1-fast-non-reasoning",
                            store_messages=False
                        )
                        chat.append(system(SYSTEM_PROMPT))
                        chat.append(user(build_user_prompt(labels)))
                        response = chat.sample()
                        text = response.content.strip()

                        dataset.append({
                            "text": text,
                            "labels": {
                                "email_address": labels[0],
                                "address": labels[1],
                                "drugs": labels[2],
                                "bioweapon": labels[3]
                            }
                        })
                        counter += 1
                        pbar.update(1)
                    except Exception as e:
                        print(f"Error generating sample: {e}")
                        continue
                    
                    if counter % save_every == 0:
                        with open(save_path, "w", encoding="utf-8") as f:
                            json.dump(dataset, f, indent=2)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2)
        print(f"Final save completed: {len(dataset)} samples written to {save_path}.")
        return dataset

    # -----------------------------
    # Label combinations
    # -----------------------------
    label_patterns = [
        [0, 0, 0, 0], [1, 1, 1, 1],
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [1, 0, 0, 1],
        [1, 0, 1, 0], [0, 1, 0, 1],
        [1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1],
    ]

    # -----------------------------
    # Execute Generation with Timestamps
    # -----------------------------
    
    date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Starting generation for date: {date_str}...")
    
    train_file = f"submission/data/{date_str}-train_dataset.json"
    val_file   = f"submission/data/{date_str}-val_dataset.json"
    test_file  = f"submission/data/{date_str}-test_dataset.json"

    generate_synthetic_dataset(client, label_patterns, train_samples, save_path=train_file)
    generate_synthetic_dataset(client, label_patterns, val_samples, save_path=val_file)
    generate_synthetic_dataset(client, label_patterns, test_samples, save_path=test_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic data using Grok API.")
    
    # Add arguments with defaults (defaults are set to the full assignment values)
    parser.add_argument("--train", type=int, default=250, help="Number of samples per pattern for training")
    parser.add_argument("--val", type=int, default=75, help="Number of samples per pattern for validation")
    parser.add_argument("--test", type=int, default=75, help="Number of samples per pattern for testing")
    
    args = parser.parse_args()

    print(f"Configuration: Train={args.train}, Val={args.val}, Test={args.test}")
    
    generate_data(
        train_samples=args.train, 
        val_samples=args.val, 
        test_samples=args.test
    )