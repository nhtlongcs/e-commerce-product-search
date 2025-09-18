"""
Simple evaluation script that only performs evaluation on a trained model checkpoint.
"""

import os
import sys
from src.params import load_parameters
from src.model import load_model_and_tokenizer
from src.dataset import create_datasets
from src.trainer import create_trainer
from src.utils import create_run_name, setup_wandb, cleanup_wandb
from transformers import TrainingArguments


def main():
    """Main evaluation function."""
    try:
        
        # Load parameters (but we'll override training settings)
        model_args, data_args, original_training_args = load_parameters()
        original_training_args.per_device_eval_batch_size = 64
        original_training_args.bf16_full_eval = True  # Ensure FP16 is used for evaluation
        print("="*60)
        print("EVALUATION MODE")
        print(f"Task: {data_args.task_name}")
        print(f"Model: {model_args.model_name_or_path}")
        print(f"Batch size: {original_training_args.per_device_eval_batch_size}")
        print(f"Cross-validation fold: {data_args.fold_id}")
        print(f"Using BF16: {original_training_args.bf16_full_eval}")
        print("="*60)
        
        # Create new TrainingArguments specifically for evaluation
        # Use minimal settings that ensure only evaluation happens
        training_args = TrainingArguments(
            output_dir=original_training_args.output_dir if hasattr(original_training_args, 'output_dir') else "./eval-outputs/",
            per_device_eval_batch_size=getattr(original_training_args, 'per_device_eval_batch_size', 128),
            dataloader_num_workers=16,
            # fp16_full_eval=getattr(original_training_args, 'fp16_full_eval', True),
            bf16_full_eval=getattr(original_training_args, 'bf16_full_eval', True),
            dataloader_pin_memory=True,
            eval_accumulation_steps=1,
            do_train=False,
            do_eval=True,
            do_predict=False,
            report_to="none",  # Disable wandb for simplicity
            logging_dir=None,
            remove_unused_columns=False,
            metric_for_best_model="f1",
            greater_is_better=True
        )
        
        # Load model and tokenizer - the model path should point to the checkpoint
        model, tokenizer = load_model_and_tokenizer(model_args)
        
        # Create datasets (only validation set needed for evaluation)
        datasets, sentence1_str, sentence2_str = create_datasets(data_args, tokenizer)
        print(f"Created datasets with columns: {sentence1_str}, {sentence2_str}")
        
        # Verify we have validation data
        if 'validation' not in datasets or datasets['validation'] is None:
            raise ValueError("No validation dataset found. Evaluation requires validation data.")
        
        print(f"Validation dataset size: {len(datasets['validation'])}")
        
        # Create trainer for evaluation only
        trainer = create_trainer(
            model=model, 
            tokenizer=tokenizer, 
            datasets=datasets, 
            training_args=training_args
        )
        
        # Perform evaluation
        print("Starting evaluation...")
        eval_metrics = trainer.evaluate()
        
        # Save metrics to file
        metrics_file = os.path.join(training_args.output_dir, "eval_results.json")
        trainer.save_metrics("eval", eval_metrics)
        
        print("\n" + "="*60)
        print("Evaluation completed successfully!")
        print(f"Evaluation results:")
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                if 'loss' in key:
                    print(f"  {key}: {value:.6f}")
                else:
                    print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\nResults saved to: {metrics_file}")
        print("="*60)
        
        return eval_metrics
        
    except Exception as e:
        print(f"Evaluation failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
