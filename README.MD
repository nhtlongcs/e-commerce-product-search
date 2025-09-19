# CIKM 2025 Competition: Multilingual E-commerce Product Search Competition: Multilingual Query-Category and Query-Item Relevance

## Requirements
We are using uv package manager to manage the environment.
To install uv, please follow the instruction at [here](https://docs.astral.sh/uv/getting-started/installation/)

To create and activate the environment, run:
```bash
uv sync
. .venv/bin/activate
```

Download Gemma3 checkpoints from
```
https://drive.google.com/file/d/106hiq7s-7eSjVQRQKanf5aCetl6pJJ6d/view?usp=sharing

https://drive.google.com/file/d/1Kk3VpQxg1j8cwjhzzElLqUfOEArW7WGa/view?usp=drive_link
```

and unzip these into `models`

Read the [model document](models/README.md) for more details about the models setup. We have multiple checkpoints for each fold. You can choose the best checkpoint based on your validation results.


## Training

```bash
bash scripts/train.sh
```
=======
Configuration files can be found in `config/QC` and `config/QI`

```bash
bash scripts/train.sh
```
## Inference

Make sure you have run the training code first.
In the folder ./models, you should have the models 
./models/gemma-3-12b-pt 
./models/QC-stage-01/<checkpoint-folders>
./models/QC-stage-02/<checkpoint-folders>
./models/QI-stage-01/<checkpoint-folders>
./models/QI-stage-02/<checkpoint-folders>

**IMPORTANT**: For gemma-3 checkpoints, the default automodel not working. So the path must have "gemma-3" in the folder name to load the model correctly.


```bash
bash scripts/predict_QC.sh
bash scripts/predict_QI.sh
```

### Sanity Check 

To ensure the integrity of the submission files, including format, number of ids, we provide a utility script for verification. 

```bash
python utils/submission_check.py <QI|QC> <submission_file>
```

To compare the difference between two submission files for performance checking, you can use the following command:

```bash
python utils/compare2sub.py submission-archive/best_submit_QC.txt <QC-submission-file>
python utils/compare2sub.py submission-archive/best_submit_QI.txt <QI-submission-file>
```

The difference should be very small if you are using the same model checkpoints. And should be lower than 10% compared to the archived submission files.