"""Configuration settings for modular architecture."""

import os
from pathlib import Path
from typing import Dict, Any

# ============================================================================
# BASE PATHS
# ============================================================================

PROJECT_ROOT = Path(os.getcwd()).resolve()
BASE_DIR = PROJECT_ROOT / "app/"
CHROMA_STORAGE_DIR = PROJECT_ROOT / "chroma_storage"
DATABASE_DIR = PROJECT_ROOT / "database"

# ============================================================================
# EMBEDDING MODELS
# ============================================================================

EMBEDDING_MODELS: Dict[str, Dict[str, Any]] = {
    "all-MiniLM-L6-v2": {
        "name": "all-MiniLM-L6-v2",
        "description": "Current baseline - fast but limited accuracy",
        "dimensions": 384,
        "performance": "fastest",
        "accuracy": "baseline"
    },
    "bge-small-en-v1.5": {
        "name": "BAAI/bge-small-en-v1.5", 
        "description": "Light upgrade - better accuracy, still fast",
        "dimensions": 384,
        "performance": "fast",
        "accuracy": "improved"
    },
    "bge-base-en-v1.5": {
        "name": "BAAI/bge-base-en-v1.5",
        "description": "Best accuracy for CPU - top-ranked model", 
        "dimensions": 768,
        "performance": "moderate",
        "accuracy": "excellent"
    },
    "e5-base-v2": {
        "name": "intfloat/e5-base-v2",
        "description": "Multi-domain enterprise - cross-department strength",
        "dimensions": 768, 
        "performance": "moderate",
        "accuracy": "excellent"
    },
    "all-mpnet-base-v2": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "description": "Classic upgrade - widely used in production",
        "dimensions": 768,
        "performance": "moderate", 
        "accuracy": "very-good"
    }
}


class Settings:
    """Application settings."""
    
    def __init__(self):
        # Base Paths
        self.PROJECT_ROOT = PROJECT_ROOT
        self.BASE_DIR = BASE_DIR
        self.CHROMA_STORAGE_DIR = CHROMA_STORAGE_DIR
        self.DATABASE_DIR = DATABASE_DIR
        self.CONFIG_DIR = BASE_DIR / "modules/config"
        self.MODELS_DIR = PROJECT_ROOT / "models"
        self.EMBEDDINGS_MODELS_DIR = PROJECT_ROOT / "embeddings_models"
        self.SENTIMENT_ARTIFACTS_DIR = PROJECT_ROOT / "sentiment"
        self.DATA_DIR = BASE_DIR / "data"
        self.TRAINING_DATA_DIR = PROJECT_ROOT / "data"
        
        # Database settings - import config
        from .database_config import db_config
        self.DB_CONFIG = db_config
        self.USERS_DB_NAME = db_config.USERS_DB
        self.SESSIONS_DB_NAME = db_config.SESSIONS_DB
        self.CONVERSATIONS_DB_NAME = db_config.CONVERSATIONS_DB
        self.DOCUMENT_VERSIONS_DB_NAME = db_config.DOCUMENT_VERSIONS_DB
        self.DEFAULT_PERSIST_DIR = str(CHROMA_STORAGE_DIR)
        self.DEFAULT_COLLECTION_NAME = db_config.DEFAULT_COLLECTION_NAME
        self.DOCUMENTS_COLLECTION = db_config.DOCUMENTS_COLLECTION

        # JWT settings
        self.JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production-please-use-env-file")
        self.JWT_ALGORITHM = "HS256"
        self.JWT_EXPIRATION_DAYS = int(os.getenv("JWT_EXPIRATION_DAYS", "1"))
        
        # Model settings
        self.DEFAULT_MODEL_NAME = "mistral-7b-instruct-v0.2.Q3_K_M.gguf"
        
        # Embedding settings
        self.EMBEDDING_MODEL_KEY = os.getenv("EMBEDDING_MODEL_KEY", "bge-small-en-v1.5")
        self.EMBEDDING_MODEL_NAME = EMBEDDING_MODELS[self.EMBEDDING_MODEL_KEY]["name"]
        self.EMBEDDING_MODELS = EMBEDDING_MODELS
        
        # Server settings
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", 8000))
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        
        # ColabLLM settings
        self.COLABLLM_BASE_URL = os.getenv("COLABLLM_BASE_URL", "https://588e8571ead7.ngrok-free.app")
        self.COLABLLM_API_KEY = os.getenv("COLABLLM_API_KEY")


# Global settings instance
settings = Settings()