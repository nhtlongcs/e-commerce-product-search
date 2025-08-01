"""
Utilities, callbacks, metrics, and Wandb integration.
"""

import os
import numpy as np
from sklearn.metrics import f1_score
from transformers import EvalPrediction
import wandb


def compute_metrics(p: EvalPrediction):
    """Compute F1 score for evaluation."""
    preds = np.argmax(p.predictions, axis=1)
    f1 = f1_score(p.label_ids, preds, pos_label=1)
    return {"f1": f1}

def create_run_name(model_args, data_args):
    """Create a unique run name based on model and task."""
    model_name = model_args.model_name_or_path.split('/')[-1].replace('-', '_').lower()
    run_name = f"{model_name}-{data_args.task_name}-{data_args.max_seq_length}"
    if data_args.fold_id is not None:
        run_name += f"-fold{data_args.fold_id}"
    return run_name, model_name

def setup_wandb(model_args, data_args, training_args, tags=None):
    """Setup Weights & Biases tracking."""
    # Set environment variables
    os.environ['WANDB_MODE'] = 'online'
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"
    os.environ["WANDB_PROJECT"] = "analyticup-dcu"
    
    # Create run name
    run_name, model_name = create_run_name(model_args, data_args)
    # Create tags
    if tags is None:
        tags = [data_args.task_name, "baseline", model_name]
    else:
        tags.append(model_name)
    if model_args.use_lora:
        tags.append("lora")
    if data_args.fold_id is not None:
        tags.append(f"fold{data_args.fold_id}")
    
    # Initialize wandb (check if already initialized by Trainer)
    if wandb.run is None:
        wandb.init(
            project="analyticup-dcu",
            name=run_name,
            tags=tags,
            group=f"{data_args.task_name}-experiments",
            job_type="train",
            notes=f"Training {model_args.model_name_or_path} on {data_args.task_name} task" + 
                  (f" (fold {data_args.fold_id})" if data_args.fold_id is not None else ""),
            config={
                "model_name": model_args.model_name_or_path,
                "task_name": data_args.task_name,
                "max_seq_length": data_args.max_seq_length,
                "use_lora": model_args.use_lora,
                "learning_rate": training_args.learning_rate,
                "batch_size": training_args.per_device_train_batch_size,
                "num_epochs": training_args.num_train_epochs,
                "fold": data_args.fold_id
            }
        )
        print(f"Wandb initialized with run name: {run_name}")
    else:
        # Update existing wandb run with our config
        wandb.run.name = run_name
        wandb.run.tags = tags
        wandb.run.group = f"{data_args.task_name}-experiments"
        wandb.config.update({
            "model_name": model_args.model_name_or_path,
            "task_name": data_args.task_name,
            "max_seq_length": data_args.max_seq_length,
            "use_lora": model_args.use_lora,
            "learning_rate": training_args.learning_rate,
            "batch_size": training_args.per_device_train_batch_size,
            "num_epochs": training_args.num_train_epochs,
            "fold": data_args.fold_id
        })
        print(f"Updated existing wandb run with name: {run_name}")
    
    # Update training args with run name
    training_args.run_name = run_name
    
    return run_name


def save_training_config(training_args, run_name):
    """Save training configuration to file."""
    import json
    
    output_dir = os.path.join(training_args.output_dir, 'runs', run_name)
    os.makedirs(output_dir, exist_ok=True)
    
    config_path = os.path.join(output_dir, 'training_args.json')
    with open(config_path, 'w') as f:
        json.dump(json.loads(training_args.to_json_string()), f, indent=4)
    
    print(f"Training config saved to: {config_path}")


def cleanup_wandb():
    """Cleanup wandb if needed."""
    if wandb.run:
        wandb.finish()
