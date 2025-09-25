"""
Main training script with resume capability.
"""

import os
import sys
from src.params import load_parameters
from src.model import load_model_and_tokenizer
from src.dataset import create_datasets
from src.trainer import create_trainer, run_training
from src.utils import create_run_name, setup_wandb, save_training_config, cleanup_wandb
from dataclasses import asdict
import json

def main():
    """Main training function."""
    try:
        # Load parameters
        model_args, data_args, training_args = load_parameters()
        

        print("="*60)
        print(f"Starting training for task: {data_args.task_name}")
        print(f"Model: {model_args.model_name_or_path}")
        if data_args.fold_id == -1:
            print("Training full dataset without evaluation.")
        else:
            print(f"Cross-validation fold: {data_args.fold_id}")
        print(f"Using BF16: {training_args.bf16}")
        print("="*60)
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_args)
        
        # Create datasets
        datasets, sentence1_str, sentence2_str = create_datasets(data_args, tokenizer)
        print(f"Created datasets with columns: {sentence1_str}, {sentence2_str}")
        
        # Temporarily disable wandb in training_args to avoid conflicts with DeepSpeed
        original_report_to = training_args.report_to
        should_use_wandb = training_args.report_to == "wandb" or "wandb" in training_args.report_to
        if should_use_wandb:
            training_args.report_to = []  # Disable automatic wandb init by Trainer
        # if data_args.fold_id == -1:
        #     training_args.eval_strategy = 'no'
        #     datasets['validation'] = None  # Override validation dataset for full training
        # Create trainer FIRST (this will initialize DeepSpeed/Accelerator without wandb conflicts)
        trainer = create_trainer(model=model, tokenizer=tokenizer, datasets=datasets, training_args=training_args)
        print(f"EVAL STRATEGY: {training_args.eval_strategy}")
        # Now setup wandb manually and restore report_to setting
        if should_use_wandb:
            run_name = setup_wandb(model_args, data_args, training_args, tags=['cross-encoder'])
            training_args.report_to = original_report_to  # Restore original setting
        else:
            run_name, _ = create_run_name(model_args, data_args, training_args)
        print(f"report to: {training_args.report_to}")
        training_args.output_dir = os.path.join(training_args.output_dir, run_name)
        print(f"Output directory: {training_args.output_dir}")     
        save_training_config(json.loads(training_args.to_json_string()), training_args.output_dir)
        save_training_config(asdict(model_args), training_args.output_dir, filename='model_args.json')
        save_training_config(asdict(data_args), training_args.output_dir, filename='data_args.json')
        
        # Run training workflow
        results = run_training(trainer=trainer, training_args=training_args)
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        if 'train_metrics' in results:
            print(f"Final training loss: {results['train_metrics'].get('train_loss', 'N/A')}")
        if 'eval_metrics' in results:
            print(f"Final F1 score: {results['eval_metrics'].get('eval_f1', 'N/A'):.4f}")
        print("="*60)
        
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    finally:
        # Cleanup
        cleanup_wandb()


if __name__ == "__main__":
    main()
