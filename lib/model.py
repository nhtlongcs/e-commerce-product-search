"""
Model loading and configuration.
"""

import os
import re

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AutoConfig
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    PeftConfig
)
# from .gemma3 import Gemma3ForSequenceClassification
def load_model_and_tokenizer(model_args, num_labels=2):
    """Load model and tokenizer with optional PEFT configuration."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check for PEFT configuration
    config_peft = None
    if (model_args.model_name_or_path and 
        os.path.isdir(model_args.model_name_or_path) and 
        'adapter_config.json' in os.listdir(model_args.model_name_or_path)):
        config_peft = PeftConfig.from_pretrained(model_args.model_name_or_path)
    
    # Load configuration
    # hard code
    print(f"Loading configuration for {model_args.model_name_or_path}")
    if 'gemma-3' in model_args.model_name_or_path or 'gemma_3' in model_args.model_name_or_path:
        print('LOAD SPECIAL GEMMA-3')
        from transformers.models.gemma3.configuration_gemma3 import Gemma3Config
        if config_peft:
            config = Gemma3Config.from_pretrained(
                config_peft.base_model_name_or_path,
                num_labels=num_labels
            )
            model_name = config_peft.base_model_name_or_path
        else:
            config = Gemma3Config.from_pretrained(
                model_args.model_name_or_path,
                num_labels=num_labels
            )
            model_name = model_args.model_name_or_path
    else:
        if config_peft:
            config = AutoConfig.from_pretrained(
                config_peft.base_model_name_or_path,
                num_labels=num_labels
            )
            model_name = config_peft.base_model_name_or_path
        else:
            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                num_labels=num_labels
            )
            model_name = model_args.model_name_or_path
    print(f"CONFIG TYPE: {type(config)}")
    # Load tokenizer
    print(f"Loading tokenizer from: {model_name}")
    print(f"Use fast tokenizer: {model_args.use_fast_tokenizer}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=model_args.use_fast_tokenizer
    )
    
    # Set pad token if not available
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set pad_token to eos_token")
    
    # Load model
    print(f"Loading model from: {model_args.model_name_or_path}")

    if 'gemma-3' in model_args.model_name_or_path or 'gemma_3' in model_args.model_name_or_path:
        from .gemma3 import Gemma3ForSequenceClassification
        model = Gemma3ForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )

    model_modules = str(model.modules)
    pattern = r'\((\w+)\): Linear'
    linear_layer_names = list(re.findall(pattern, model_modules))

    # Apply PEFT if configured
    if model_args.use_lora:
        print("Applying LoRA configuration")
        print(f"Linear layers found: {linear_layer_names}")
        if not model_args.lora_target_modules:
            raise ValueError("lora_target_modules must be specified when use_lora=True")
        
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=model_args.lora_target_modules,
            modules_to_save=model_args.lora_modules_to_save,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.01,
        )
        model = get_peft_model(model, peft_config)
        print(f"LoRA applied with target modules: {model_args.lora_target_modules}")
    
    # Configure pad token and move to device
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    
    print("Model and tokenizer loaded successfully")
    return model, tokenizer
