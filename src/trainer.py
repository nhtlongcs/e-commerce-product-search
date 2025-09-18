"""
Training workflow management.
"""

from transformers import Trainer
from .dataset import create_data_collator
from .utils import compute_metrics


def create_trainer(model, tokenizer, training_args, datasets=None, train_dataset=None, eval_dataset=None):
    """Create Trainer instance with all necessary components."""
    
    # Create data collator
    data_collator = create_data_collator(tokenizer)
    
    # Handle different parameter formats
    if datasets is not None:
        train_ds = datasets.get('train') if training_args.do_train else None
        eval_ds = datasets.get('validation') if training_args.do_eval else None
    else:
        train_ds = train_dataset
        eval_ds = eval_dataset
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        # callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Optional
    )
    
    return trainer


def run_training(trainer, training_args):
    """Execute training workflow."""
    results = {}
    
    # Training
    if training_args.do_train:
        print("Starting training...")
        
        # Check for checkpoint
        checkpoint = None
        if hasattr(training_args, 'resume_from_checkpoint') and training_args.resume_from_checkpoint:
            checkpoint = training_args.resume_from_checkpoint
            print(f"Resuming from checkpoint: {checkpoint}")
        
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        
        # Save model and tokenizer
        trainer.save_model()
        
        # Log and save metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        results['train_metrics'] = metrics
        print("Training completed successfully")
    
    # Evaluation
    if training_args.do_eval:
        print("Starting evaluation...")
        
        # Evaluate
        eval_metrics = trainer.evaluate()
        
        # Log and save metrics
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)
        
        results['eval_metrics'] = eval_metrics
        print(f"Evaluation completed. F1 score: {eval_metrics.get('eval_f1', 'N/A'):.4f}")
    
    return results
