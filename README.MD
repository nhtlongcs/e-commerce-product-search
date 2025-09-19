# CIKM 2025 Competition: Multilingual E-commerce Product Search Competition: Multilingual Query-Category and Query-Item Relevance
> Recent advances in large language models (LLMs) have opened new opportunities for improving multilingual search in e-commerce platforms, where queries are often noisy, code-mixed, and expressed in low-resource languages. This report presents our approach for the Alibaba AIDC Multilingual Product Search Challenge, which consists of two tasks: Query-Category (QC) and Query-Item (QI) relevance classification. We adopt a systematic methodology combining careful data preprocessing, category-aware cross-validation splitting, and query augmentation through English translation and concatenation. Our experiments benchmark a range of multilingual LLMs, including Qwen and Gemma series, and apply parameter-efficient fine-tuning via LoRA with DeepSpeed-enabled mixed-precision training. We further enhance performance through ensemble inference across cross-validation folds. 


Team: DcuRAGONS - Dublin City University, Dublin, Ireland
Members:
- Thang-Long Nguyen Ho: thanglong.nguyenho27@mail.dcu.ie
- Hoang-Bao Le: bao.le2@mail.dcu.ie
- Minh-Khoi Pham: minhkhoi.pham4@mail.dcu.ie

Technical report can be found in `report/`

## Requirements
We are using uv package manager to manage the environment.
To install uv, please follow the instruction at [here](https://docs.astral.sh/uv/getting-started/installation/)

To create and activate the environment, run:
```bash
uv sync
. .venv/bin/activate
```

Download Gemma3 checkpoints from
- [Gemma3-stage1](https://drive.google.com/file/d/106hiq7s-7eSjVQRQKanf5aCetl6pJJ6d/view?usp=sharing)
- [Gemma3-stage2](https://drive.google.com/file/d/1Kk3VpQxg1j8cwjhzzElLqUfOEArW7WGa/view?usp=drive_link)

and unzip these into `models`
Correct file path to load these models should be as follow: `models/best-gemma-3-QC-stage-01`, ...


Read the [model document](models/README.md) for more details about the models setup. We have multiple checkpoints for each fold. You can choose the best checkpoint based on your validation results.


## Data preprocessing

Data preprocess steps includes: cross-validation splitting and translation

```bash
bash scripts/preprocess.sh
```
Final files we use for training are: `data/translated/translated_train_QI_full_fold` and `data/translated/translated_train_QC_full_fold_v2`


## Training

```bash
bash scripts/train.sh
```

Configuration files can be found in `config/QC` and `config/QI`

## Inference

Make sure you have run the training code first.
In the folder ./models, you should have the models' paths as follow:
```
./models/gemma-3-12b-pt 
./models/QC-stage-01/<checkpoint-folders>
./models/QC-stage-02/<checkpoint-folders>
./models/QI-stage-01/<checkpoint-folders>
./models/QI-stage-02/<checkpoint-folders>
```

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

The submission files of our team for the preliminary round are in `submission-archive` folder, named as `best_submit_<TASK>.txt`