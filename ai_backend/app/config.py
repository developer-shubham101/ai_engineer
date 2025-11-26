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

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# --- LLM ---
ENABLE_DYNAMIC_MODEL_SELECTION = False  # Set to True to enable dynamic model selection based on task
DEFAULT_MODEL_NAME = "mistral-7b-instruct-v0.2.Q3_K_M.gguf"  # Primary model to use

# --- Embeddings ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


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
