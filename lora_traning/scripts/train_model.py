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
        result = await train_company_model(
            output_name="distilgpt2-company-tuned",
            max_samples=500,  # Smaller for testing
            epochs=2,
            learning_rate=2e-5
        )

        logger.info("Training completed successfully!")
        logger.info(f"Model saved to: {result['model_path']}")
        if result.get('gguf_path'):
            logger.info(f"GGUF file: {result['gguf_path']}")

    except Exception as e:
        logger.exception("Training failed: %s", e)


if __name__ == "__main__":
    asyncio.run(main())
