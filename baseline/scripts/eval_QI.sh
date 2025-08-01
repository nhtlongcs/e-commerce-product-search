#!/bin/bash

# Evaluation script for QI task
echo "Starting QI evaluation..."

# Set task
TASK="QI"

# Run evaluation with config files
python eval.py \
    config/${TASK}/data.json \
    config/${TASK}/model.json \
    config/${TASK}/eval.json

echo "QI evaluation completed!"
