# app/services/local_model_manager.py
"""
Local model management service for multiple LLM options.
Handles model detection, selection, and auto-downloading.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import os

logger = logging.getLogger(__name__)

# Paths
def get_config_path(filename: str) -> Path:
    return Path("app/config") / filename

def get_models_path() -> Path:
    return Path("models")

class LocalModelManager:
    """Manages multiple local LLM models with auto-detection and downloading."""
    
    def __init__(self):
        self.config_path = get_config_path("local_models.json")
        self.models_dir = get_models_path()
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
                        "size_mb": round(model_path.stat().st_size / (1024*1024), 1),
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