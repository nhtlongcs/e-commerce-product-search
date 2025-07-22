import json
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    EvalPrediction,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftConfig
)
import torch
from torch.utils.data import DataLoader

from dataset import SentencePairDataset, SentencePairPredictDataset
from parameters import load_parameters

os.environ['WANDB_MODE'] = 'offline'
# os.environ['TRITON_PTXAS_PATH']='/usr/local/cuda/bin/ptxas'


# Configuration
model_args, data_args, training_args = load_parameters()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Function to compute metrics
def compute_metrics(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, preds, pos_label=1)
    return {"f1": f1}


# Load config
config_peft = None
if model_args.model_name_or_path is not None and os.path.isdir(model_args.model_name_or_path) and 'adapter_config.json' in os.listdir(model_args.model_name_or_path):
    config_peft = PeftConfig.from_pretrained(
        model_args.model_name_or_path
    )
    config = AutoConfig.from_pretrained(
        config_peft.base_model_name_or_path,
        num_labels=2
    )
else:
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        num_labels=2
    )

# Load tokenizer and model
print("use fast tokenizer = {}".format(model_args.use_fast_tokenizer))
tokenizer = AutoTokenizer.from_pretrained(
    config_peft.base_model_name_or_path if isinstance(config_peft, PeftConfig) else model_args.model_name_or_path,
    use_fast=model_args.use_fast_tokenizer
)
if not hasattr(tokenizer, 'pad_token'):
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token

model = AutoModelForSequenceClassification.from_pretrained(
    model_args.model_name_or_path,
    num_labels=2,
    problem_type="single_label_classification"
)
if model_args.use_lora:
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=model_args.lora_target_modules,
        modules_to_save=model_args.lora_modules_to_save,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )
    model = get_peft_model(model, peft_config)
model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)


# Create datasets
if data_args.task_name == 'QC':
    sentence1_str = 'origin_query'
    sentence2_str = 'category_path'
if data_args.task_name == 'QI':
    sentence1_str = 'origin_query'
    sentence2_str = 'item_title'

train_dataset = SentencePairDataset(data_args.train_file, tokenizer, data_args.max_seq_length, sentence1_str, sentence2_str)
eval_dataset = SentencePairDataset(data_args.validation_file, tokenizer, data_args.max_seq_length, sentence1_str, sentence2_str)
test_dataset = SentencePairPredictDataset(data_args.test_file, tokenizer, data_args.max_seq_length, sentence1_str, sentence2_str)
data_collator = DataCollatorWithPadding(tokenizer, padding="longest")


# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if training_args.do_train else None,
    eval_dataset=eval_dataset if training_args.do_eval else None,
    compute_metrics=compute_metrics,
    data_collator = data_collator
    # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)


# Train the model
if training_args.do_train:
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    print("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


# Evaluate the model
if training_args.do_eval:
    print("Evaluating on validation set...")
    metrics = trainer.evaluate(eval_dataset)
    
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    print(f"Final F1 score on test set: {metrics['eval_f1']:.4f}")


# Predict
if training_args.do_predict:
    dataloader = DataLoader(test_dataset, batch_size=training_args.per_device_eval_batch_size, shuffle=False)

    with open(data_args.outputs, 'w', encoding='utf-8') as f:
        for i, batch in enumerate(tqdm(dataloader)):
            # Tokenize inputs
            inputs = tokenizer(
                batch[sentence1_str], batch[sentence2_str],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=data_args.max_seq_length
            ).to(model.device)

            # Predict
            with torch.no_grad():
                outputs = model(**inputs)

            # Get prediction (0 or 1)
            prediction = torch.argmax(outputs.logits, dim=1).tolist()

            # Create output JSON
            for k in range(len(batch['id'])):
                output_json = {
                    "id": batch["id"][k].item(),
                    "language": batch["language"][k],
                    sentence1_str: batch[sentence1_str][k],
                    sentence2_str: batch[sentence2_str][k],
                    "prediction": prediction[k]
                }

                # Write to file
                f.write(json.dumps(output_json, ensure_ascii=False) + "\n")

    print(f"Predictions saved to {data_args.outputs}")
