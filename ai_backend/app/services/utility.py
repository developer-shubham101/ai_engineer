"""
Centralized utility module for common paths, constants, and shared functions.
This module prevents code duplication and circular import issues.
"""
import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)

# ============================================================================
# BASE PATHS
# ============================================================================

PROJECT_ROOT = Path(os.getcwd()).resolve()

BASE_DIR = PROJECT_ROOT / "app/"
# PROJECT_ROOT is the directory above 'app/'


# ============================================================================
# MODEL CONSTANTS
# ============================================================================
from app.config import EMBEDDING_MODEL_NAME

# ============================================================================
# DIRECTORY PATHS
# ============================================================================
# DATA_DIR (relative to app/) is app/data - for app-specific data
DATA_DIR = BASE_DIR / "data"
# DATABASE_DIR (relative to project root) is project_root/database - for SQLite databases
from app.config import DATABASE_DIR
# TRAINING_DATA_DIR (relative to project root) is project_root/data
TRAINING_DATA_DIR = PROJECT_ROOT / "data"
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_MODELS_DIR = PROJECT_ROOT / "embeddings_models"
SENTIMENT_ARTIFACTS_DIR = PROJECT_ROOT / "sentiment"

# ============================================================================
# CHROMA DEFAULTS
# ============================================================================
from app.config import DEFAULT_PERSIST_DIR, DEFAULT_COLLECTION_NAME


# ============================================================================
# FILE PATHS
# ============================================================================
def get_local_embedding_model_path() -> Path:
    """Get the path to the local embedding model directory."""
    return EMBEDDINGS_MODELS_DIR / EMBEDDING_MODEL_NAME


def get_config_path(filename: str) -> Path:
    """Get path to a config file."""
    return CONFIG_DIR / filename


def get_data_path(filename: str) -> Path:
    """Get path to a data file. This resolves to PROJECT_ROOT/data/filename."""
    return TRAINING_DATA_DIR / filename


def get_sentiment_artifact_path(filename: str) -> Path:
    """Get path to a sentiment classifier artifact file."""
    return SENTIMENT_ARTIFACTS_DIR / filename


def get_database_path(filename: str) -> Path:
    """Get path to a database file in the database directory."""
    return DATABASE_DIR / filename


# ============================================================================
# EMBEDDING MODEL LOADER (Singleton to prevent duplicate loading)
# ============================================================================
_embedding_model_instance = None


def get_embedding_model_instance():
    """
    Get or create a singleton SentenceTransformer instance.
    This prevents loading the same model multiple times.
    """
    global _embedding_model_instance

    if _embedding_model_instance is not None:
        return _embedding_model_instance

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence_transformers not installed. "
            "Install sentence-transformers to compute local embeddings."
        )

    local_path = get_local_embedding_model_path()
    if local_path.exists():
        logger.info("Loading embedding model from local path: %s", local_path)
        _embedding_model_instance = SentenceTransformer(str(local_path))
    else:
        logger.info(
            "Loading embedding model by name (may download if not cached): %s",
            EMBEDDING_MODEL_NAME
        )
        _embedding_model_instance = SentenceTransformer(EMBEDDING_MODEL_NAME)

    return _embedding_model_instance


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using the shared embedding model."""
    model = get_embedding_model_instance()
    vectors = await run_in_threadpool(model.encode, texts, convert_to_numpy=True)
    return vectors.tolist()


# ============================================================================
# TEXT PROCESSING
# ============================================================================
def chunk_text_basic(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Produce overlapping chunks of the input text.
    Fixed so we always make progress and produce expected overlaps.
    """
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        if end == L:
            break
        # advance start keeping overlap, but ensure progress by at least 1
        start = max(end - overlap, start + 1)
    return chunks


# ============================================================================
# METADATA SANITIZATION
# ============================================================================
def sanitize_meta_value(val):
    """
    Ensure metadata values are primitives (str, int, float, bool) for Chroma.
    - If val is list of primitives -> join with commas
    - If val is dict -> json.dumps
    - Else convert to str
    """
    import json
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, list):
        # if list of primitives, join; otherwise json-dump
        if all(isinstance(x, (str, int, float, bool)) for x in val):
            return ",".join(str(x) for x in val)
        return json.dumps(val, ensure_ascii=False)
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    # fallback
    return str(val)


def sanitize_metadata_dict(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Sanitize a metadata dictionary for Chroma compatibility."""
    if not meta:
        return {}
    return {str(k): sanitize_meta_value(v) for k, v in meta.items()}








# ============================================================================
# BACKWARD COMPATIBILITY (for existing code)
# ============================================================================
# Keep the old function name for backward compatibility
def _get_local_embedding_model_path() -> Path:
    """Backward compatibility alias."""
    return get_local_embedding_model_path()


def is_empty(data):
    # Case 1: None
    if data is None:
        return True

    # Case 2: Iterable types (list, dict, tuple, set, string)
    if isinstance(data, (list, dict, tuple, set, str)):
        return len(data) == 0

    # Case 3: Has length
    if hasattr(data, "__len__"):
        return len(data) == 0

    return True


def is_collection_empty(data):
    """Return True if Chroma/Vector DB response represents an empty collection."""
    if data is None:
        return True

    # If dictionary structure
    if isinstance(data, dict):
        # Chroma empty pattern: ids == [] and documents == []
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])

        # "Empty" means: all primary fields contain no usable data
        if len(ids) == 0 and len(docs) == 0 and len(metas) == 0:
            return True

        return False  # some data exists

    # If some vector store object
    if hasattr(data, "ids"):
        ids = getattr(data, "ids", [])
        return len(ids) == 0

    if hasattr(data, "documents"):
        docs = getattr(data, "documents", [])
        return len(docs) == 0

    # Generic fallback
    try:
        return len(data) == 0
    except Exception:
        return not bool(data)
