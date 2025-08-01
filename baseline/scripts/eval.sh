#!/bin/bash

# Generic evaluation script
# Usage: ./eval.sh [TASK] [CHECKPOINT_PATH]
# Example: ./eval.sh QI ./QI-outputs/checkpoint-3189

TASK=${1:-QI}
CHECKPOINT_PATH=${2:-""}

echo "Starting ${TASK} evaluation..."

# Set checkpoint in eval config if provided
if [ ! -z "$CHECKPOINT_PATH" ]; then
    echo "Using checkpoint: $CHECKPOINT_PATH"
    # Create temporary eval config with custom checkpoint
    TEMP_CONFIG="config/${TASK}/eval_temp.json"
    cat config/${TASK}/eval.json | jq --arg checkpoint "$CHECKPOINT_PATH" '.resume_from_checkpoint = $checkpoint' > $TEMP_CONFIG
    
    # Run evaluation with temporary config
    python eval.py \
        config/${TASK}/data.json \
        config/${TASK}/model.json \
        $TEMP_CONFIG
    
    # Clean up temporary config
    rm $TEMP_CONFIG
else
    # Run evaluation with default config
    python eval.py \
        config/${TASK}/data.json \
        config/${TASK}/model.json \
        config/${TASK}/eval.json
fi

echo "${TASK} evaluation completed!"
