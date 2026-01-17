#!/usr/bin/env python3
"""
Standalone script to train DistilGPT2 on company data.
Can be run independently of the API server.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add app to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.model_training_service import train_company_model, is_training_available
from app.config.model_config import ModelConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Train model with company data."""
    if not is_training_available():
        logger.error("Training dependencies not available. Install with:")
        logger.error("pip install -r requirements.txt")
        return

    try:
        logger.info("Starting model training...")
        config = ModelConfig.get_model_config()
        result = await train_company_model(
            output_name=config["output_name"],
            max_samples=config["max_samples"],
            epochs=config["epochs"],
            learning_rate=config["learning_rate"],
            dataset_name=config.get("dataset_name") # Pass HF dataset name
        )

        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {result['model_path']}")
        if result.get('gguf_path'):
            logger.info(f"GGUF file: {result['gguf_path']}")

    except Exception as e:
        logger.exception("Training failed: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
