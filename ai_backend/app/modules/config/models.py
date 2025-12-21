"""Configuration data models."""

from typing import Dict, Any, Optional, List, Set
from pydantic import BaseModel
from dataclasses import dataclass
from .database_config import db_config
from ..config.settings import settings


@dataclass
class ModelConfig:
    """Local model configuration."""
    name: str
    file_path: str
    context_length: int
    description: str
    parameters: Dict[str, Any]


@dataclass
class ProviderConfig:
    """LLM provider configuration."""
    name: str
    api_key_required: bool
    models: List[str]
    default_model: str
    max_tokens: int
    supports_temperature: bool


class DatabaseConfig(BaseModel):
    """Database configuration."""
    users_db: str = db_config.USERS_DB
    sessions_db: str = db_config.SESSIONS_DB
    versions_db: str = db_config.DOCUMENT_VERSIONS_DB
    chroma_collection: str = db_config.DOCUMENTS_COLLECTION


class SecurityConfig(BaseModel):
    """Security configuration."""
    jwt_secret_key: str
    jwt_algorithm: str = settings.JWT_ALGORITHM
    jwt_expiration_days: int = 7
    password_min_length: int = 6
    max_login_attempts: int = 5


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model_name: str = settings.EMBEDDING_MODEL_KEY
    model_path: Optional[str] = None
    dimension: int = 384
    batch_size: int = 32


class RAGConfig(BaseModel):
    """RAG system configuration."""
    default_top_k: int = 3
    max_top_k: int = 20
    default_temperature: float = 0.1
    max_context_length: int = 2048
    chunk_size: int = 500
    chunk_overlap: int = 50


class ValidationConfig(BaseModel):
    """Validation configuration."""
    max_file_size_mb: int = 5
    supported_extensions: List[str] = [".txt", ".md", ".markdown", ".html", ".htm", ".json", ".csv"]
    hr_level_threshold: int = 2


class AppConfig(BaseModel):
    """Complete application configuration."""
    database: DatabaseConfig = DatabaseConfig()
    security: SecurityConfig = SecurityConfig(jwt_secret_key="change-me")
    embedding: EmbeddingConfig = EmbeddingConfig()
    rag: RAGConfig = RAGConfig()
    validation: ValidationConfig = ValidationConfig()
    debug: bool = False
    log_level: str = "INFO"