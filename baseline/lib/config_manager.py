"""
Configuration manager for loading and combining separate config files.
"""

import json
import os
from typing import Dict, Optional


class ConfigManager:
    """Manages loading and combining configuration files."""
    
    def __init__(self, config_base_path: str = "config"):
        self.config_base_path = config_base_path
    
    def load_config(self, config_path: str) -> Dict:
        """Load a single configuration file."""
        if not os.path.isabs(config_path):
            config_path = os.path.join(self.config_base_path, config_path)
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def create_training_config(self, 
                             data_config_path: str, 
                             model_config_path: str, 
                             training_config_path: str,
                             output_path: Optional[str] = None) -> Dict:
        """Create a combined training configuration from separate config files."""
        data_config = self.load_config(data_config_path)
        model_config = self.load_config(model_config_path)
        training_config = self.load_config(training_config_path)
        
        # Combine configs (training config takes precedence for overlapping keys)
        combined_config = {}
        combined_config.update(data_config)
        combined_config.update(model_config)
        combined_config.update(training_config)
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(combined_config, f, indent=2)
        
        return combined_config
    
    def create_prediction_config(self, 
                               data_config_path: str, 
                               model_config_path: str,
                               checkpoint_path: str,
                               output_path: Optional[str] = None) -> Dict:
        """Create a prediction configuration from data and model configs."""
        data_config = self.load_config(data_config_path)
        model_config = self.load_config(model_config_path)
        
        # Combine configs and update model path to checkpoint
        combined_config = {}
        combined_config.update(data_config)
        combined_config.update(model_config)
        combined_config["model_name_or_path"] = checkpoint_path
        
        # Add evaluation batch size if not present
        if "per_device_eval_batch_size" not in combined_config:
            combined_config["per_device_eval_batch_size"] = 8
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(combined_config, f, indent=2)
        
        return combined_config
    
    def create_fold_config(self, 
                          base_config: Dict, 
                          fold_num: int,
                          output_path: Optional[str] = None) -> Dict:
        """Create fold-specific configuration."""
        fold_config = base_config.copy()
        fold_config["fold"] = fold_num
        
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(fold_config, f, indent=2)
        
        return fold_config


# Convenience functions for backward compatibility
def create_training_config_file(task: str, model_type: str = "base", output_path: str = None):
    """Create a training configuration file for a given task and model type."""
    config_manager = ConfigManager()
    
    data_config_path = f"data/data_{task}.json"
    model_config_path = f"model/model_{model_type}.json"
    training_config_path = f"training/training_{task}.json"
    
    if output_path is None:
        suffix = "_XLMR" if model_type == "xlmr" else ""
        output_path = f"train_{task}{suffix}.json"
    
    return config_manager.create_training_config(
        data_config_path, model_config_path, training_config_path, output_path
    )


def create_prediction_config_file(task: str, checkpoint_path: str, model_type: str = "base", output_path: str = None):
    """Create a prediction configuration file for a given task and checkpoint."""
    config_manager = ConfigManager()
    
    data_config_path = f"data/data_{task}.json"
    model_config_path = f"model/model_{model_type}.json"
    
    if output_path is None:
        suffix = "_XLMR" if model_type == "xlmr" else ""
        output_path = f"predict/predict_{task}{suffix}.json"
    
    return config_manager.create_prediction_config(
        data_config_path, model_config_path, checkpoint_path, output_path
    )
