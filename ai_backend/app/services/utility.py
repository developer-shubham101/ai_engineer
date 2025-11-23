"""
Centralized utility module for common paths, constants, and shared functions.
This module prevents code duplication and circular import issues.
"""
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# BASE PATHS
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent  # app/
PROJECT_ROOT = BASE_DIR.parent  # project root

# ============================================================================
# MODEL CONSTANTS
# ============================================================================
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ============================================================================
# DIRECTORY PATHS
# ============================================================================
DATA_DIR = BASE_DIR / "data"
CONFIG_DIR = BASE_DIR / "config"
MODELS_DIR = PROJECT_ROOT / "models"
EMBEDDINGS_MODELS_DIR = PROJECT_ROOT / "embeddings_models"
CHROMA_STORAGE_DIR = BASE_DIR / "chroma_storage"
SENTIMENT_ARTIFACTS_DIR = PROJECT_ROOT / "sentiment"

# ============================================================================
# CHROMA DEFAULTS
# ============================================================================
DEFAULT_PERSIST_DIR = CHROMA_STORAGE_DIR
DEFAULT_COLLECTION_NAME = "local_manual_rag"

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
    """Get path to a data file."""
    return DATA_DIR / filename


def get_sentiment_artifact_path(filename: str) -> Path:
    """Get path to a sentiment classifier artifact file."""
    return SENTIMENT_ARTIFACTS_DIR / filename


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


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using the shared embedding model."""
    model = get_embedding_model_instance()
    vectors = model.encode(texts, convert_to_numpy=True).tolist()
    return vectors


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
# TONE NORMALIZATION
# ============================================================================
# Canonical tone labels: angry, confused, happy, frustrated, polite, urgent, neutral
CANONICAL_TONES = {"angry", "confused", "happy", "frustrated", "polite", "urgent", "neutral"}


def normalize_tone_label(raw_tone: Optional[str]) -> str:
    """
    Map raw model output tone labels to canonical tones.
    Returns one of: angry, confused, happy, frustrated, polite, urgent, neutral
    Defaults to 'neutral' if tone is None or unrecognized.
    """
    if not raw_tone:
        return "neutral"
    
    tone_lower = raw_tone.lower().strip()
    
    # Direct match
    if tone_lower in CANONICAL_TONES:
        return tone_lower
    
    # Mapping rules for common variations
    tone_mapping = {
        # Angry variants
        "furious": "angry",
        "annoyed": "angry",
        "irritated": "angry",
        "mad": "angry",
        # Happy variants
        "appreciative": "happy",
        "pleased": "happy",
        "satisfied": "happy",
        "grateful": "happy",
        # Frustrated variants
        "upset": "frustrated",
        "disappointed": "frustrated",
        "stressed": "frustrated",
        # Polite variants
        "curious": "polite",
        "respectful": "polite",
        "courteous": "polite",
        # Urgent variants
        "emergency": "urgent",
        "critical": "urgent",
        "asap": "urgent",
        # Confused variants
        "uncertain": "confused",
        "unclear": "confused",
        "lost": "confused",
    }
    
    # Check mapping
    if tone_lower in tone_mapping:
        return tone_mapping[tone_lower]
    
    # Partial matching for compound tones
    for key, canonical in tone_mapping.items():
        if key in tone_lower or tone_lower in key:
            return canonical
    
    # Default fallback
    logger.debug("Unrecognized tone label '%s', defaulting to 'neutral'", raw_tone)
    return "neutral"


# ============================================================================
# TONE GUIDANCE
# ============================================================================
def build_tone_guidance(tone: Optional[str]) -> str:
    """
    Map a tone label into a short LLM instruction for tone-sensitive replies.
    Keep these instructions concise (20-60 tokens) — they are injected into the model prompt.
    Uses normalized canonical tones.
    """
    # Normalize tone to canonical form
    normalized = normalize_tone_label(tone)
    
    # Short, token-efficient guidance for each canonical tone
    guidance_map = {
        "angry": "User is angry. Respond calmly, acknowledge frustration, focus on fast resolution.",
        "frustrated": "User is frustrated. Be patient, provide step-by-step guidance.",
        "confused": "User is confused. Simplify explanation, use examples.",
        "urgent": "User indicates urgency. Prioritize actionable steps, be direct.",
        "happy": "User is positive. Respond warmly and helpfully.",
        "polite": "User is polite. Respond normally with clear information.",
        "neutral": "Respond normally in a helpful and friendly manner.",
    }
    
    return guidance_map.get(normalized, guidance_map["neutral"])


# ============================================================================
# BACKWARD COMPATIBILITY (for existing code)
# ============================================================================
# Keep the old function name for backward compatibility
def _get_local_embedding_model_path() -> Path:
    """Backward compatibility alias."""
    return get_local_embedding_model_path()
