                            LICENSE & USAGE
Copyright (c) 2026 Nancy N. Wei. All rights reserved.

This repository is source-available for portfolio review only.
No permission is granted to use, copy, modify, merge, publish, distribute,
sublicense, sell, or create derivative works from this project, in whole or in
part, without prior written permission from the author.

See the root LICENSE file for full terms.



STEP 1. INSTALLATION
--------------------
Install the libraries as listed in requirements.txt:
    pip install -r requirements.txt


STEP 2. DATA GENERATION
-----------------------
a) TINY SAMPLE (Fast Test)
Run the following to generate a dataset with 64 samples (32 training, 16 validation, 16 test):
    python content_safety_classifier/data/data_generation.py --train 2 --val 1 --test 1

b) FULL DATASET (High Performance)
Only run this if you have sufficient storage and ~3 hours. 
This generates 6,400 samples (4,000 training, 1,200 validation, 1,200 test):
    python submission/data/data_generation.py --train 250 --val 75 --test 75

*Note: The larger dataset significantly improves the fine-tuned model's accuracy.*


STEP 3. TRAINING
----------------
First, find the timestamp of your generated data:
    ls content_safety_classifier/data/

Then, run the training script:
    python content_safety_classifier/model/train.py \
      --train_file content_safety_classifier/data/TIMESTAMP-train_dataset.json \
      --val_file content_safety_classifier/data/TIMESTAMP-val_dataset.json \
      --test_file content_safety_classifier/data/TIMESTAMP-test_dataset.json

*Replace TIMESTAMP with the actual date/time string from your data files.*

The script will:
1. Load the data.
2. Fine-tune the Qwen/Qwen2.5-1.5B-Instruct model.
3. Print initial Test Metrics.
4. Save the model to: content_safety_classifier/model/checkpoint/TIMESTAMP-model/


STEP 4. INFERENCE & EVALUATION
------------------------------
Run the following to generate detailed predictions and evaluation metrics:
    python content_safety_classifier/model/inference.py \
      --model_path content_safety_classifier/model/checkpoint/TIMESTAMP-model \
      --test_file content_safety_classifier/data/TIMESTAMP-test_dataset.json

*Replace TIMESTAMP with the appropriate timestamps from your model and data.*

Outputs will be saved in the 'content_safety_classifier/evaluation' folder:
- results.json: Summary metrics (Accuracy, F1, False Positive/Negative Rates).
- inference_results.json: Detailed predictions for every test sample.
