#!/usr/bin/env python3
"""
Example script showing how to change the default model.
This demonstrates the centralized configuration approach.
"""

import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.model_config import ModelConfig

def show_current_config():
    """Show current model configuration."""
    print("Current Model Configuration:")
    print("=" * 40)
    print(f"Base Model: {ModelConfig.DEFAULT_BASE_MODEL}")
    print(f"Output Name: {ModelConfig.DEFAULT_OUTPUT_NAME}")
    print(f"Epochs: {ModelConfig.DEFAULT_EPOCHS}")
    print(f"Learning Rate: {ModelConfig.DEFAULT_LEARNING_RATE}")
    print(f"Max Samples: {ModelConfig.DEFAULT_MAX_SAMPLES}")
    print()

def show_example_configs():
    """Show example configurations for different models."""
    print("Example Model Configurations:")
    print("=" * 40)
    
    examples = [
        ("distilgpt2", "Fast, lightweight model (default)"),
        ("gpt2", "Standard GPT-2 model"),
        ("microsoft/DialoGPT-small", "Dialog-focused model"),
        ("microsoft/DialoGPT-medium", "Larger dialog model"),
    ]
    
    for model, description in examples:
        config = ModelConfig.get_model_config(model)
        print(f"Model: {model}")
        print(f"Description: {description}")
        print(f"Output Name: {config['output_name']}")
        print(f"Base Model: {config['base_model']}")
        print()

def main():
    """Main function."""
    print("Model Configuration Manager")
    print("=" * 50)
    print()
    
    show_current_config()
    show_example_configs()
    
    print("To change the default model:")
    print("1. Edit app/config/model_config.py")
    print("2. Update DEFAULT_BASE_MODEL and DEFAULT_OUTPUT_NAME")
    print("3. All scripts will automatically use the new configuration")
    print()
    print("Example changes:")
    print('   DEFAULT_BASE_MODEL = "gpt2"')
    print('   DEFAULT_OUTPUT_NAME = "gpt2-company-tuned"')

if __name__ == "__main__":
    main()