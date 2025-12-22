#!/usr/bin/env python3
"""
Simple script to run the AI Backend application.
"""

import uvicorn
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    print("🚀 Starting AI Backend with Modular Architecture")
    print("=" * 50)
    from app.modules.config.settings import settings
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True,
        log_level="info"
    )