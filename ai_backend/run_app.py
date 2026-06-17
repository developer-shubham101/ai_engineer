#!/usr/bin/env python3
"""
Simple script to run the AI Backend application.
"""

import logging
import sys
from pathlib import Path

import uvicorn

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    logging.info("Starting AI Backend with Modular Architecture")
    from app.modules.config.settings import settings
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        reload_excludes=[
            "*.log",
            "logs/*",
            "logs/**/*",
        ],
        log_level="info",
    )
