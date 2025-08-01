#!/bin/bash
if [ "$1" = "train" ]; then
    torchrun --nproc_per_node 4 --nnodes=1 train.py config/QI/data.json config/QI/model.json config/QI/train.json
elif [ "$1" = "predict" ]; then
    python predict.py config/QI/data.json config/QI/model.json
elif [ "$1" = "eval" ]; then
    python eval.py ./QI-outputs/checkpoint-3189/ config/QI/data.json config/QI/model.json
else
    echo "Usage: $0 {train|predict|eval}"
    exit 1
fi