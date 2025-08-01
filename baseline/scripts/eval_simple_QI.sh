#!/bin/bash

# Simple evaluation script for QI task
echo "Starting QI evaluation (simple)..."

# Set task
TASK="QI"

# Run evaluation with config files
python eval_simple.py \
    config/${TASK}/data.json \
    config/${TASK}/model.json

echo "QI evaluation completed!"
