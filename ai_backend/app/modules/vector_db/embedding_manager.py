"""Embedding manager implementation."""

import logging
from typing import List, Dict, Any
from pathlib import Path
import os

from sentence_transformers import SentenceTransformer
from starlette.concurrency import run_in_threadpool

from .interfaces import IEmbeddingManager
from ..config.settings import settings

logger = logging.getLogger(__name__)

_embedding_model_instance = None

class EmbeddingManager(IEmbeddingManager):
    """Singleton embedding manager implementation."""
    
    def __init__(self):
        global _embedding_model_instance
        if _embedding_model_instance is None:
            _embedding_model_instance = self._load_model()
        self._model = _embedding_model_instance

    def _load_model(self):
        """
        Get or create a singleton SentenceTransformer instance.
        Enhanced embedding model with improved logging and error handling.
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            logger.error("EMBEDDING_MODEL_ERROR: sentence_transformers not installed")
            raise ImportError(
                "sentence_transformers not installed. "
                "Install sentence-transformers to compute local embeddings."
            )

        model_config = settings.EMBEDDING_MODELS.get(settings.EMBEDDING_MODEL_KEY, {})
        logger.info(
            "EMBEDDING_MODEL_INIT: Loading model '%s' (key: %s)", 
            settings.EMBEDDING_MODEL_NAME, settings.EMBEDDING_MODEL_KEY
        )
        logger.info(
            "EMBEDDING_MODEL_CONFIG: %s | Dimensions: %s | Performance: %s | Accuracy: %s",
            model_config.get("description", "Unknown"),
            model_config.get("dimensions", "Unknown"),
            model_config.get("performance", "Unknown"), 
            model_config.get("accuracy", "Unknown")
        )

        local_path = settings.EMBEDDINGS_MODELS_DIR / settings.EMBEDDING_MODEL_KEY
        
        try:
            if local_path.exists():
                logger.info("EMBEDDING_MODEL_LOCAL: Loading from local path: %s", local_path)
                model = SentenceTransformer(str(local_path))
            else:
                logger.info(
                    "EMBEDDING_MODEL_DOWNLOAD: Loading by name (may download): %s",
                    settings.EMBEDDING_MODEL_NAME
                )
                model = SentenceTransformer(settings.EMBEDDING_MODEL_NAME)
            
            model_info = model.get_sentence_embedding_dimension()
            logger.info(
                "EMBEDDING_MODEL_SUCCESS: Model loaded successfully | Actual dimensions: %d", 
                model_info
            )
            
            expected_dims = model_config.get("dimensions")
            if expected_dims and model_info != expected_dims:
                logger.warning(
                    "EMBEDDING_MODEL_DIMENSION_MISMATCH: Expected %d dimensions, got %d",
                    expected_dims, model_info
                )
            return model
        except Exception as e:
            logger.error(
                "EMBEDDING_MODEL_LOAD_FAILED: Failed to load model '%s': %s", 
                settings.EMBEDDING_MODEL_NAME, str(e)
            )
            if settings.EMBEDDING_MODEL_KEY != "all-MiniLM-L6-v2":
                logger.warning("EMBEDDING_MODEL_FALLBACK: Attempting fallback to MiniLM")
                try:
                    model = SentenceTransformer("all-MiniLM-L6-v2")
                    logger.info("EMBEDDING_MODEL_FALLBACK_SUCCESS: Using MiniLM as fallback")
                    return model
                except Exception as fallback_error:
                    logger.error("EMBEDDING_MODEL_FALLBACK_FAILED: %s", str(fallback_error))
                    raise
            else:
                raise
    
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings."""
        import time
        
        # Performance monitoring
        start_time = time.time()
        text_count = len(texts)
        total_chars = sum(len(text) for text in texts)
        
        logger.debug(
            "EMBEDDING_ENCODE_START: Processing %d texts, %d total characters", 
            text_count, total_chars
        )
        
        try:
            vectors = await run_in_threadpool(self._model.encode, texts, convert_to_numpy=True)
            
            elapsed = time.time() - start_time
            chars_per_sec = total_chars / elapsed if elapsed > 0 else 0
            
            logger.debug(
                "EMBEDDING_ENCODE_SUCCESS: %d vectors generated in %.2fs (%.0f chars/sec)",
                len(vectors), elapsed, chars_per_sec
            )
            
            return vectors.tolist()
            
        except Exception as e:
            logger.error(
                "EMBEDDING_ENCODE_FAILED: Error encoding %d texts: %s", 
                text_count, str(e)
            )
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get embedding model information."""
        model_config = settings.EMBEDDING_MODELS.get(settings.EMBEDDING_MODEL_KEY, {})
        
        info = {
            "model_key": settings.EMBEDDING_MODEL_KEY,
            "model_name": settings.EMBEDDING_MODEL_NAME,
            "description": model_config.get("description", "Unknown"),
            "dimensions": model_config.get("dimensions", "Unknown"),
            "performance": model_config.get("performance", "Unknown"),
            "accuracy": model_config.get("accuracy", "Unknown"),
            "local_path": str(settings.EMBEDDINGS_MODELS_DIR / settings.EMBEDDING_MODEL_KEY),
            "local_exists": (settings.EMBEDDINGS_MODELS_DIR / settings.EMBEDDING_MODEL_KEY).exists()
        }
        
        if self._model is not None:
            try:
                info["actual_dimensions"] = self._model.get_sentence_embedding_dimension()
                info["model_loaded"] = True
            except Exception as e:
                info["model_loaded"] = False
                info["load_error"] = str(e)
        else:
            info["model_loaded"] = False
            
        return info

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        if self._model:
            return self._model.get_sentence_embedding_dimension()
        return 0
