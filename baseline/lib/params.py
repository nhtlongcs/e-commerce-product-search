"""
Parameters and configuration management.
"""

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from transformers import HfArgumentParser, TrainingArguments


@dataclass
class DataArguments:
    """Arguments for data configuration."""
    task_name: str = field(default='QI', metadata={"help": "Task: QC or QI"})
    max_seq_length: int = field(default=128, metadata={"help": "Maximum sequence length"})
    train_file: Optional[str] = field(default=None, metadata={"help": "Training data file"})
    validation_file: Optional[str] = field(default=None, metadata={"help": "Validation data file"})
    test_file: Optional[str] = field(default=None, metadata={"help": "Test data file"})
    outputs: str = field(default="submit.json", metadata={"help": "Output predictions file"})
    fold_id: Optional[int] = field(default=None, metadata={"help": "Cross-validation fold (0, 1, 2)"})

    def get_sentence_columns(self, task_name: Optional[str] = None):
        """Get sentence column names based on task."""
        if task_name is None:
            task_name = self.task_name

        if task_name == 'QC':
            return 'origin_query', 'category_path'
        elif task_name == 'QI':
            return 'origin_query', 'item_title'
        elif task_name == 'QC_en':
            return 'translated_query', 'category_path'
        elif task_name == 'QI_en':
            return 'translated_query', 'item_title'
        elif task_name == 'QC_2lang':
            return 'origin_query,translated_query', 'category_path'
        elif task_name == 'QI_2lang':
            return 'origin_query,translated_query', 'item_title'
        else:
            raise ValueError(f"Unknown task: {task_name}")


@dataclass
class ModelArguments:
    """Model configuration arguments."""
    model_name_or_path: str = field(metadata={"help": "Model name or path"})
    use_fast_tokenizer: bool = field(default=True, metadata={"help": "Use fast tokenizer"})
    use_lora: bool = field(default=False, metadata={"help": "Use LoRA PEFT"})
    lora_target_modules: str = field(default=None, metadata={"help": "LoRA target modules (comma-separated)"})
    lora_modules_to_save: str = field(default=None, metadata={"help": "LoRA modules to save (comma-separated)"})

    def __post_init__(self):
        if self.lora_target_modules:
            self.lora_target_modules = self.lora_target_modules.split(',')
        if self.lora_modules_to_save:
            self.lora_modules_to_save = self.lora_modules_to_save.split(',')


def load_parameters():
    """Load parameters from command line or JSON file."""
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    
    if len(sys.argv) >= 2 and all(arg.endswith(".json") for arg in sys.argv[1:]):
        # Load and merge multiple JSON files
        config_dict = {}
        for json_file in sys.argv[1:]:
            with open(os.path.abspath(json_file), 'r') as f:
                config_part = json.load(f)
                config_dict.update(config_part)

        if 'output_dir' not in config_dict:
            config_dict['output_dir'] = './temp_output'

        model_args, data_args, training_args = parser.parse_dict(config_dict)
    elif len(sys.argv) >= 3 and not sys.argv[1].endswith(".json") and all(arg.endswith(".json") for arg in sys.argv[2:]):
        # Overwrite model_name_or_path from command line, load configs from JSON files
        model_name_or_path = sys.argv[1]
        config_dict = {}
        for json_file in sys.argv[2:]:
            with open(os.path.abspath(json_file), 'r') as f:
                config_part = json.load(f)
                config_dict.update(config_part)
        config_dict['model_name_or_path'] = model_name_or_path

        if 'output_dir' not in config_dict:
            config_dict['output_dir'] = './temp_output'

        model_args, data_args, training_args = parser.parse_dict(config_dict)
    else:
        # Parse from command line
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    return model_args, data_args, training_args


def load_prediction_parameters():
    """Load parameters for prediction (no training config needed)."""
    parser = HfArgumentParser((ModelArguments, DataArguments))
    
    if len(sys.argv) >= 2 and all(arg.endswith(".json") for arg in sys.argv[1:]):
        # Load and merge multiple JSON files
        config_dict = {}
        for json_file in sys.argv[1:]:
            with open(os.path.abspath(json_file), 'r') as f:
                config_part = json.load(f)
                config_dict.update(config_part)
        
        model_args, data_args = parser.parse_dict(config_dict)
    else:
        # Parse from command line
        model_args, data_args = parser.parse_args_into_dataclasses()
    
    return model_args, data_args


def create_training_config(data_config: dict, model_config: dict, training_config: dict) -> dict:
    """Combine separate configs into a single training config."""
    combined_config = {}
    combined_config.update(data_config)
    combined_config.update(model_config)
    combined_config.update(training_config)
    return combined_config


def create_prediction_config(data_config: dict, model_config: dict) -> dict:
    """Combine data and model configs for prediction (no training config)."""
    combined_config = {}
    combined_config.update(data_config)
    combined_config.update(model_config)
    return combined_config


def create_fold_config(config_file: str, fold_num: int) -> str:
    """Create fold-specific config file."""
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    config['fold'] = fold_num
    fold_config_file = config_file.replace('.json', f'_fold_{fold_num}.json')
    
    with open(fold_config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    return fold_config_file
