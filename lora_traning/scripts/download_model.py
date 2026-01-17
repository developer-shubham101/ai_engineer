#!/usr/bin/env python3
"""
Script to download the base model and save it locally.
This ensures we have a local copy of the model for training.
"""

import sys
import logging
from pathlib import Path
import shutil

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.model_config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    logger.error("Transformers not installed. Run: pip install transformers torch")
    sys.exit(1)

def download_model():
    """Download base model defined in config to local models directory."""
    config = ModelConfig.get_model_config()
    base_model_name = config["base_model"]
    
    # Sanitize model name for directory path
    sanitized_name = base_model_name.replace("/", "-")
    
    # Define local path
    root_dir = Path(__file__).parent.parent
    models_dir = root_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    local_model_path = models_dir / sanitized_name
    
    if local_model_path.exists():
        logger.info(f"Model directory already exists at: {local_model_path}")
        logger.info("If you want to re-download, delete this directory first.")
        return

    logger.info(f"Downloading model: {base_model_name}")
    logger.info(f"Destination: {local_model_path}")
    
    try:
        # Download tokenizer
        logger.info("Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        tokenizer.save_pretrained(local_model_path)
        
        # Download model
        logger.info("Downloading model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            device_map="auto", # Download to CPU/GPU automatically
            trust_remote_code=True
        )
        model.save_pretrained(local_model_path)
        
        logger.info("Successfully downloaded model and tokenizer!")
        logger.info(f"Local path: {local_model_path}")
        
    except Exception as e:
        logger.exception(f"Failed to download model: {e}")
        # Cleanup partial download
        if local_model_path.exists():
            shutil.rmtree(local_model_path)
            
if __name__ == "__main__":
    download_model()
