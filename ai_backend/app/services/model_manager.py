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

from app.config import ENABLE_DYNAMIC_MODEL_SELECTION, DEFAULT_MODEL_NAME
from app.services.utility import MODELS_DIR

logger = logging.getLogger(__name__)

# Internal global handles - model cache
_llm_instances: Dict[str, Any] = {}  # cache for different model keys


def choose_model_for_task(task: str) -> str:
    """
    Choose appropriate model for a given task.

    This function routes tasks to different model sizes ("tiny", "small", "mistral")
    based on the task description. This allows for using smaller, faster models for
    simple tasks and larger, more powerful models for complex reasoning.

    Returns:
        "tiny" - for short chit-chat, quick responses
        "small" - for summarization, classification, tagging, intent detection
        "mistral" - for full RAG reasoning (default)

    Args:
        task: Task type - "chat", "summarize", "classify", "tag", "intent", "reason", "rag", etc.
    """
    task_lower = task.lower()

    # Small model tasks
    if task_lower in ["summarize", "classification", "classify", "tag", "tagging", "intent", "intent_detection"]:
        return "small"

    # Tiny model tasks
    if task_lower in ["chat", "chit-chat", "quick", "simple"]:
        return "tiny"

    # Default to mistral for RAG, reasoning, and unknown tasks
    return "mistral"


def get_llm_instance(model_key: str = "default"):
    """
    Lazy-load and cache LLM instances.

    This function manages the loading and caching of local LLM instances.
    It first checks a cache to see if the model is already loaded. If not,
    it searches for the model file on disk and initializes a LlamaCpp instance.
    It supports both a default model and dynamic selection of models based on a key.

    By default, uses the specific model: mistral-7b-instruct-v0.2.Q3_K_M.gguf
    If ENABLE_DYNAMIC_MODEL_SELECTION is True and default model not found,
    falls back to dynamic selection based on model_key patterns.

    Args:
        model_key: Model identifier (only used if dynamic selection enabled)
                  "mistral" - Full RAG model
                  "small" - Smaller model for summarization/classification
                  "tiny" - Smallest model for quick chat

    Returns:
        LlamaCpp instance (cached)
    """
    global _llm_instances

    # Check cache first (use "default" as cache key for primary model)
    cache_key = "default"
    if model_key in _llm_instances:
        logger.debug("Returning cached LLM instance for key=%s", model_key)
        return _llm_instances[model_key]
    if cache_key in _llm_instances:
        logger.debug("Returning cached LLM instance for cache_key=%s", cache_key)
        return _llm_instances[cache_key]

    if LlamaCpp is None:
        raise RuntimeError("llama-cpp-python not installed. Install llama-cpp-python to use local LLM.")

    model_path = None
    config = {"n_ctx": 2048, "n_batch": 8}  # Default config for mistral

    # First, try to find the specific default model
    default_model_path = MODELS_DIR / DEFAULT_MODEL_NAME
    if default_model_path.exists():
        model_path = str(default_model_path)
        logger.info("Found default model: %s", model_path)
    else:
        # Try with different extensions
        for ext in [".gguf", ".ggml", ".bin"]:
            test_path = MODELS_DIR / (DEFAULT_MODEL_NAME.rsplit(".", 1)[0] + ext)
            if test_path.exists():
                model_path = str(test_path)
                logger.info("Found default model (with %s extension): %s", ext, model_path)
                break

    # If default model not found AND dynamic selection is enabled, do dynamic search
    if not model_path and ENABLE_DYNAMIC_MODEL_SELECTION:
        logger.info("Default model not found. Dynamic selection enabled. Searching by model_key='%s'", model_key)

        # Model file patterns by key
        model_patterns = {
            "mistral": ["*mistral*.gguf", "*mistral*.ggml", "*mistral*.bin"],
            "small": ["*small*.gguf", "*small*.ggml", "*small*.bin", "*7b*.gguf", "*7b*.ggml"],
            "tiny": ["*tiny*.gguf", "*tiny*.ggml", "*tiny*.bin", "*1b*.gguf", "*1b*.ggml", "*3b*.gguf", "*3b*.ggml"],
        }

        # Model configs by key (n_ctx, n_batch)
        model_configs = {
            "mistral": {"n_ctx": 2048, "n_batch": 8},
            "small": {"n_ctx": 1024, "n_batch": 4},
            "tiny": {"n_ctx": 512, "n_batch": 2},
        }

        patterns = model_patterns.get(model_key, model_patterns["mistral"])
        config = model_configs.get(model_key, model_configs["mistral"])

        # Search for model file using patterns
        for pattern in patterns:
            files = list(MODELS_DIR.glob(pattern))
            if files:
                model_path = str(files[0])
                logger.info("Found model via dynamic search: %s (pattern: %s)", model_path, pattern)
                break

        # Last resort: find any model file
        if not model_path:
            for ext in ("*.gguf", "*.ggml", "*.bin"):
                files = list(MODELS_DIR.glob(ext))
                if files:
                    model_path = str(files[0])
                    logger.warning("Using fallback model via dynamic search: %s", model_path)
                    break

    if not model_path:
        if ENABLE_DYNAMIC_MODEL_SELECTION:
            raise RuntimeError(
                f"No model file found in {MODELS_DIR}. "
                f"Expected '{DEFAULT_MODEL_NAME}' or dynamic selection patterns."
            )
        else:
            raise RuntimeError(
                f"Default model '{DEFAULT_MODEL_NAME}' not found in {MODELS_DIR}. "
                f"Set ENABLE_DYNAMIC_MODEL_SELECTION=True to enable dynamic model selection."
            )

    logger.info("Loading LlamaCpp model: path=%s, n_ctx=%d, n_batch=%d", model_path, config["n_ctx"], config["n_batch"])
    instance = LlamaCpp(
        model_path=model_path,
        n_ctx=config["n_ctx"],
        n_batch=config["n_batch"],
        n_gpu_layers=0  # CPU-only
    )

    # Cache the instance (use "default" as key for primary model)
    _llm_instances[cache_key] = instance
    return instance
