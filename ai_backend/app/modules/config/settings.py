"""Application settings and configuration management."""

import os
from typing import Optional, Dict, Any
from pathlib import Path
import json


class Settings:
    """Application settings manager."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent.parent.parent
        self._load_environment()
        self._load_model_configs()
    
    def _load_environment(self):
        """Load environment variables."""
        # Server settings
        self.HOST = os.getenv("HOST", "0.0.0.0")
        self.PORT = int(os.getenv("PORT", 8000))
        self.DEBUG = os.getenv("DEBUG", "false").lower() == "true"
        
        # API Keys (optional)
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        self.HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
        
        # JWT settings
        self.JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-change-in-production")
        self.JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
        self.JWT_EXPIRATION_DAYS = int(os.getenv("JWT_EXPIRATION_DAYS", 7))
        
        # Model settings
        self.DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL_NAME", "mistral-7b-instruct-v0.2")
        self.EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "bge-small-en-v1.5")
        
        # Database paths
        self.DATABASE_DIR = self.base_dir / "database"
        self.CHROMA_STORAGE_DIR = self.base_dir / "chroma_storage"
        self.MODELS_DIR = self.base_dir / "models"
        self.EMBEDDINGS_DIR = self.base_dir / "embeddings_models"
        
        # Ensure directories exist
        for dir_path in [self.DATABASE_DIR, self.CHROMA_STORAGE_DIR, self.MODELS_DIR, self.EMBEDDINGS_DIR]:
            dir_path.mkdir(exist_ok=True)
    
    def _load_model_configs(self):
        """Load model configurations."""
        config_path = self.base_dir / "app" / "config" / "local_models.json"
        try:
            with open(config_path, 'r') as f:
                self.LOCAL_MODELS_CONFIG = json.load(f)
        except FileNotFoundError:
            self.LOCAL_MODELS_CONFIG = {}
        
        onboarding_path = self.base_dir / "app" / "config" / "onboarding_fields.json"
        try:
            with open(onboarding_path, 'r') as f:
                self.ONBOARDING_FIELDS = json.load(f)
        except FileNotFoundError:
            self.ONBOARDING_FIELDS = {}
    
    def get_database_url(self, db_name: str) -> str:
        """Get database URL for given database name."""
        return f"sqlite:///{self.DATABASE_DIR / db_name}"
    
    def has_api_key(self, provider: str) -> bool:
        """Check if API key is available for provider."""
        key_map = {
            "openai": self.OPENAI_API_KEY,
            "google": self.GOOGLE_API_KEY,
            "huggingface": self.HUGGINGFACE_API_TOKEN
        }
        return bool(key_map.get(provider.lower()))
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for provider."""
        key_map = {
            "openai": self.OPENAI_API_KEY,
            "google": self.GOOGLE_API_KEY,
            "huggingface": self.HUGGINGFACE_API_TOKEN
        }
        return key_map.get(provider.lower())


# Global settings instance
settings = Settings()