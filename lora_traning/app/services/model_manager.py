# app/services/model_manager.py
"""
This module handles the loading and management of local LLM instances.
"""
import logging
from pathlib import Path
from typing import Dict, Any

try:
    from langchain.llms import LlamaCpp
except Exception:
    LlamaCpp = None

from app.services.local_model_manager import get_model_manager

# Constants
DEFAULT_MODEL_NAME = "distilgpt2-company-tuned.gguf"
MODELS_DIR = Path("models")

logger = logging.getLogger(__name__)

# Internal global handles - model cache
_llm_instances: Dict[str, Any] = {}  # cache for different model keys





class ModelManager:
    """Simple model manager for GGUF models."""
    
    def __init__(self):
        self.models_dir = Path("models")
        self.model_cache = {}
    
    def load_model(self, model_name: str):
        """Load a GGUF model."""
        if model_name in self.model_cache:
            return self.model_cache[model_name]
        
        # Try to find GGUF file
        gguf_path = self.models_dir / f"{model_name}.gguf"
        if gguf_path.exists():
            try:
                from llama_cpp import Llama
                model = Llama(
                    model_path=str(gguf_path),
                    n_ctx=2048,
                    n_threads=4,
                    verbose=False
                )
                self.model_cache[model_name] = model
                return model
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                return None
        
        return None

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
        default_path = MODELS_DIR / DEFAULT_MODEL_NAME
        if default_path.exists():
            model_path = str(default_path)
            logger.info("Using default model file: %s", DEFAULT_MODEL_NAME)
        else:
            raise RuntimeError(f"No models available. Expected '{DEFAULT_MODEL_NAME}' or models from local_models.json")
    
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
