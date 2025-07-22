# CIKM 2025 Competition: Multilingual E-commerce Product Search Competition: Multilingual Query-Category and Query-Item Relevance

## Run the Baseline

### Data

Download the data and put it in the folder .data/
```tree
.
├── data
│   ├── train_QC.txt
│   ├── train_QI.txt
│   ├── dev_QC.txt
│   └── dev_QI.txt
```
In the final round, dev_QI.txt and dev_QC.txt should be replaced by test_QC.txt and test_QI.txt

### Training

```shell
# Run the training
./scripts/train.sh

# Run the training for the category prediction (QC) task
./scripts/train_QC.sh

# Run the training for the query-item relevance (QI) task
./scripts/train_QI.sh
```

### Predict

Make sure you have run the training code first.
In the folder ./models, you should have the models ./models/baseline_QC and ./models/baseline_QI

```shell
# Run the prediction
./scripts/predict.sh

# Run the prediction for the category prediction (QC) task
./scripts/predict_QC.sh

# Run the prediction for the query-item relevance (QI) task
./scripts/predict_QI.sh
```

### Submit the results
After running the baseline, the prediction results should be in the folder .predictions/

```tree
.
├── predictions
│   ├── submit_QC.txt
│   └── submit_QI.txt
```

Zip the two files in submit.zip. You can use the following command:
```shell
# Compress the prediction files
./scripts/compress_submit.sh
```

Then, upload submit.zip to the platform.

### Submit your code
If you need to submit your code, it should be submitted in a zip file, and reproducible.
- It should have a train.sh and a predict.sh files that enables to reproduce the training and prediction parts, in the script/ folder.
- Modifications of data, or new data, should be included in the folder data/
- A README should contain additional explanations on how to reproduce the results.
- Besides, a requirements.txt should be included
- The final model should be in the folder model/