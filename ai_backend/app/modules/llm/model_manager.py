# app/modules/llm/model_manager.py
"""
Local model management service for multiple LLM options.
Handles model detection, selection, and auto-downloading.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

from langchain_community.llms import LlamaCpp

from ..config.settings import settings

logger = logging.getLogger(__name__)

# Internal global handles - model cache
_llm_instances: Dict[str, Any] = {}  # cache for different model keys


class LocalModelManager:
    """Manages multiple local LLM models with auto-detection and downloading."""

    def __init__(self):
        self.config_path = settings.CONFIG_DIR / "local_models.json"
        self.models_dir = settings.MODELS_DIR
        self.config = self._load_config()
        self._available_models = None

    def _load_config(self) -> Dict[str, Any]:
        """Load model configuration from JSON file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load model config: {e}")
            return {"models": {}, "default_model": "mistral-7b"}

    def get_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available (downloaded) models."""
        if self._available_models is not None:
            return self._available_models

        available = {}
        models_config = self.config.get("models", {})

        for model_key, model_info in models_config.items():
            gguf_file = model_info.get("gguf_file")
            if gguf_file:
                model_path = self.models_dir / gguf_file
                if model_path.exists():
                    available[model_key] = {
                        **model_info,
                        "path": str(model_path),
                        "size_mb": round(model_path.stat().st_size / (1024 * 1024), 1),
                        "available": True
                    }
                else:
                    available[model_key] = {
                        **model_info,
                        "path": None,
                        "available": False
                    }

        self._available_models = available
        return available

    def get_default_model(self) -> str:
        """Get the default model key."""
        return self.config.get("default_model", "phi2")

    def get_model_info(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific model."""
        available = self.get_available_models()
        return available.get(model_key)

    def get_model_path(self, model_key: str) -> Optional[Path]:
        """Get the file path for a model if it exists."""
        model_info = self.get_model_info(model_key)
        if model_info and model_info.get("available"):
            return Path(model_info["path"])
        return None

    def list_downloadable_models(self) -> List[Dict[str, Any]]:
        """List models that can be downloaded."""
        available = self.get_available_models()
        downloadable = []

        for model_key, model_info in available.items():
            if not model_info.get("available") and model_info.get("download_url"):
                downloadable.append({
                    "key": model_key,
                    "name": model_info.get("name"),
                    "size": model_info.get("size"),
                    "description": model_info.get("description"),
                    "download_url": model_info.get("download_url"),
                    "gguf_file": model_info.get("gguf_file")
                })

        return downloadable

    def get_best_available_model(self) -> Optional[str]:
        """Get the best available model (default first, then any available)."""
        available = self.get_available_models()

        # Try default model first
        default_model = self.get_default_model()
        if default_model in available and available[default_model].get("available"):
            return default_model

        # Try recommended models
        for model_key, model_info in available.items():
            if model_info.get("available") and model_info.get("recommended"):
                return model_key

        # Return any available model
        for model_key, model_info in available.items():
            if model_info.get("available"):
                return model_key

        return None

    def refresh_cache(self):
        """Refresh the cached available models list."""
        self._available_models = None


# Global instance
_model_manager = LocalModelManager()


def get_model_manager() -> LocalModelManager:
    """Get the global model manager instance."""
    return _model_manager


def get_llm_instance(model_key: str = None):
    """
    Lazy-load and cache LLM instances with dynamic model selection.
    
    Args:
        model_key: Model key from local_models.json (e.g., "llama32-1b", "phi2")
                  If None, uses best available model or falls back to default
    
    Returns:
        LlamaCpp instance (cached)
    """
    global _llm_instances

    cache_key = model_key or "default"
    if cache_key in _llm_instances:
        logger.debug("Returning cached LLM instance for key=%s", cache_key)
        return _llm_instances[cache_key]

    if LlamaCpp is None:
        raise RuntimeError("llama-cpp-python not installed. Install llama-cpp-python to use local LLM.")

    model_manager = get_model_manager()
    model_path = None
    context_length = 2048

    # Try specific model key first
    if model_key:
        model_path = model_manager.get_model_path(model_key)
        if model_path:
            model_info = model_manager.get_model_info(model_key)
            context_length = model_info.get("context_length", 2048)
            logger.info("Using requested model: %s", model_key)
        else:
            logger.warning("Requested model '%s' not available, falling back", model_key)

    # Fallback to best available model
    if not model_path:
        best_model_key = model_manager.get_best_available_model()
        if best_model_key:
            model_path = model_manager.get_model_path(best_model_key)
            model_info = model_manager.get_model_info(best_model_key)
            context_length = model_info.get("context_length", 2048)
            logger.info("Using best available model: %s", best_model_key)

    # Final fallback to default model file
    if not model_path:
        default_path = settings.MODELS_DIR / settings.DEFAULT_MODEL_NAME
        if default_path.exists():
            model_path = str(default_path)
            logger.info("Using default model file: %s", settings.DEFAULT_MODEL_NAME)
        else:
            raise RuntimeError(
                f"No models available. Expected '{settings.DEFAULT_MODEL_NAME}' or models from local_models.json")

    config = {"n_ctx": context_length, "n_batch": 8}

    logger.info("Loading LlamaCpp model: %s (n_ctx=%d)", model_path, context_length)
    instance = LlamaCpp(
        model_path=str(model_path),
        n_ctx=config["n_ctx"],
        n_batch=config["n_batch"],
        n_gpu_layers=0  # CPU-only
    )

    _llm_instances[cache_key] = instance
    return instance
