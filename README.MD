# CIKM 2025 Competition: Multilingual E-commerce Product Search Competition: Multilingual Query-Category and Query-Item Relevance

## Requirements
We are using uv package manager to manage the environment.
To install uv, please follow the instruction at https://uv.readthedocs.io/en/latest

To create and activate the environment, run:
```
uv sync
. .venv/bin/activate
```

## Run the Baseline


### Predict

Make sure you have run the training code first.
In the folder ./models, you should have the models 
./models/gemma-3-12b-pt 
./models/QC-stage-01/<checkpoint-folders>
./models/QC-stage-02/<checkpoint-folders>
./models/QI-stage-01/<checkpoint-folders>
./models/QI-stage-02/<checkpoint-folders>

We have multiple checkpoints for each fold. You can choose the best checkpoint based on your validation results.

```bash
bash scripts/predict_QC.sh
bash scripts/predict_QI.sh
```