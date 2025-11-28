# app/config.py
"""
Centralized configuration for the application.
"""
from pathlib import Path
import os

# ============================================================================
# BASE PATHS
# ============================================================================

PROJECT_ROOT = Path(os.getcwd()).resolve()
BASE_DIR = PROJECT_ROOT / "app/"
CHROMA_STORAGE_DIR = PROJECT_ROOT / "chroma_storage"
DATABASE_DIR = PROJECT_ROOT / "database"

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# --- LLM ---
DEFAULT_MODEL_NAME = "mistral-7b-instruct-v0.2.Q3_K_M.gguf"  # Primary model to use

# --- Embeddings ---
# Embedding model upgrade path:
# Phase 1: bge-small-en-v1.5 (light upgrade from MiniLM)
# Phase 2: BAAI/bge-base-en-v1.5 (best accuracy for CPU)
# Phase 3: intfloat/e5-base-v2 (multi-domain enterprise)

# Available embedding models (all CPU-friendly, 768 dimensions)
EMBEDDING_MODELS = {
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

# Current active model (recommended: start with bge-small-en-v1.5 for now set as default)
EMBEDDING_MODEL_KEY = os.getenv("EMBEDDING_MODEL_KEY", "all-MiniLM-L6-v2")
EMBEDDING_MODEL_NAME = EMBEDDING_MODELS[EMBEDDING_MODEL_KEY]["name"]


# ============================================================================
# CHROMA DB CONFIGURATION
# ============================================================================
DEFAULT_PERSIST_DIR = str(CHROMA_STORAGE_DIR)
DEFAULT_COLLECTION_NAME = "local_manual_rag"

# ============================================================================
# AUTHENTICATION & SECURITY
# ============================================================================
# JWT Configuration
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production-please-use-env-file")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_DAYS = int(os.getenv("JWT_EXPIRATION_DAYS", "1"))  # Token expires in 1 day by default
