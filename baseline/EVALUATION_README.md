# Evaluation Scripts

This directory contains evaluation scripts for loading trained model checkpoints and evaluating their performance.

## Files Created

### Main Evaluation Scripts
- `eval.py` - Full-featured evaluation script with wandb support and checkpoint auto-detection
- `eval_simple.py` - Simplified evaluation script with minimal dependencies

### Configuration Files
- `config/QI/eval.json` - Evaluation configuration for QI task
- `scripts/eval_QI.sh` - Shell script to run QI evaluation
- `scripts/eval.sh` - Generic evaluation script for any task
- `scripts/eval_simple_QI.sh` - Simple evaluation script for QI task

## Usage

### Method 1: Full Evaluation Script (with wandb)
```bash
# Run QI evaluation with specific configs
python eval.py config/QI/data.json config/QI/model.json config/QI/eval.json

# Or use the shell script
./scripts/eval_QI.sh

# Or use the generic script
./scripts/eval.sh QI ./QI-outputs/checkpoint-3189
```

### Method 2: Simple Evaluation Script (no wandb)
```bash
# Run simple evaluation
python eval_simple.py config/QI/data.json config/QI/model.json

# Or use the shell script
./scripts/eval_simple_QI.sh
```

## Key Features

### eval.py
- ✅ Loads checkpoints from specified path or auto-detects latest checkpoint
- ✅ Supports wandb logging (can be disabled)
- ✅ Uses DeepSpeed if configured
- ✅ Comprehensive error handling
- ✅ Overrides training parameters to ensure no training occurs
- ✅ Saves evaluation metrics to file

### eval_simple.py  
- ✅ Minimal dependencies (no wandb)
- ✅ Clean TrainingArguments for evaluation only
- ✅ Automatic checkpoint loading from model config
- ✅ Simple and fast execution

## Configuration

### Key Parameters Overridden for Evaluation
- `do_train: false` - Disables training
- `do_eval: true` - Enables evaluation
- `do_predict: false` - Disables prediction
- `resume_from_checkpoint` - Points to the checkpoint to load

### Checkpoint Loading
The scripts will load checkpoints in this priority order:
1. `resume_from_checkpoint` in config
2. Model path specified in `model_name_or_path` 
3. Latest checkpoint in `output_dir`

## Output
- Evaluation metrics printed to console
- Metrics saved to `eval_results.json` in output directory
- wandb logging (if enabled in full version)

## Example Output
```
============================================================
EVALUATION MODE
Task: QI
Model: ./QI-outputs/checkpoint-3189
Cross-validation fold: 0
============================================================
Created datasets with columns: origin_query, item_title
Validation dataset size: 12345
Starting evaluation...

============================================================
Evaluation completed successfully!
Evaluation results:
  eval_loss: 0.234567
  eval_f1: 0.8765
  eval_precision: 0.8654
  eval_recall: 0.8876
  eval_accuracy: 0.8543
  eval_runtime: 45.23
  eval_samples_per_second: 273.15
============================================================
```
