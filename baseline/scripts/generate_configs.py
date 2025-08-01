#!/usr/bin/env python3
"""
Script to generate combined configuration files from modular config components.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.config_manager import ConfigManager


def generate_training_configs():
    """Generate all training configuration files."""
    config_manager = ConfigManager("config")
    
    configs = [
        # (task, model_type, output_file)
        ("QI", "base", "config/train_QI.json"),
        ("QC", "base", "config/train_QC.json"),
        ("QI", "xlmr", "config/train_QI_XLMR.json"),
        ("QC", "xlmr", "config/train_QC_XLMR.json"),
    ]
    
    for task, model_type, output_file in configs:
        print(f"Generating {output_file}...")
        
        data_config_path = f"data/data_{task}.json"
        model_config_path = f"model/model_{model_type}.json"
        training_config_path = f"training/training_{task}.json"
        
        config_manager.create_training_config(
            data_config_path, model_config_path, training_config_path, output_file
        )


def generate_cv_configs():
    """Generate cross-validation configuration files."""
    config_manager = ConfigManager("config")
    
    configs = [
        # (task, model_type, base_config)
        ("QI", "base", "config/train_QI.json"),
        ("QC", "base", "config/train_QC.json"),
    ]
    
    for task, model_type, base_config in configs:
        # Load base config
        base_config_dict = config_manager.load_config(base_config)
        
        # Generate fold configs
        for fold in range(3):
            output_file = f"config/train_{task}_cv_fold_{fold}.json"
            print(f"Generating {output_file}...")
            config_manager.create_fold_config(base_config_dict, fold, output_file)


def main():
    """Main function to generate all configuration files."""
    print("Generating training configurations...")
    generate_training_configs()
    
    print("\nGenerating cross-validation configurations...")
    generate_cv_configs()
    
    print("\nAll configuration files generated successfully!")
    print("\nTraining configs:")
    print("- config/train_QI.json")
    print("- config/train_QC.json") 
    print("- config/train_QI_XLMR.json")
    print("- config/train_QC_XLMR.json")
    
    print("\nPrediction configs:")
    print("- config/predict/predict_QI.json")
    print("- config/predict/predict_QC.json")
    print("- config/predict/predict_QI_XLMR.json")
    print("- config/predict/predict_QC_XLMR.json")
    
    print("\nModular configs:")
    print("- config/data/data_QI.json")
    print("- config/data/data_QC.json")
    print("- config/model/model_base.json")
    print("- config/model/model_xlmr.json")
    print("- config/training/training_QI.json")
    print("- config/training/training_QC.json")


if __name__ == "__main__":
    main()
