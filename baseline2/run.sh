#!/bin/bash
if [ "$1" = "train" ]; then
    torchrun --nproc_per_node 4 --nnodes=1 train.py config/QI/data.json config/QI/model.json config/QI/train.json
elif [ "$1" = "predict" ]; then
    python predict.py config/QI/data.json config/QI/model.json
elif [ "$1" = "eval" ]; then
    CKPT=$2
    if [ -z "$CKPT" ]; then
        echo "Checkpoint path is required for evaluation."
        exit 1
    fi
    python eval.py "$CKPT" config/QI/data.json
else
    echo "Usage: $0 {train|predict|eval}"
    exit 1
fi

torchrun --nproc_per_node 4 --nnodes=1 train.py config/QI/data.json config/QI/model.json config/QI/train.json