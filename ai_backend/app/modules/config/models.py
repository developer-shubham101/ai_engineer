"""Configuration data models."""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from dataclasses import dataclass


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
    users_db: str = "users.db"
    sessions_db: str = "support_sessions.db"
    versions_db: str = "document_versions.db"
    chroma_collection: str = "rag_documents"


class SecurityConfig(BaseModel):
    """Security configuration."""
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expiration_days: int = 7
    password_min_length: int = 6
    max_login_attempts: int = 5


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""
    model_name: str = "bge-small-en-v1.5"
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


class AppConfig(BaseModel):
    """Complete application configuration."""
    database: DatabaseConfig = DatabaseConfig()
    security: SecurityConfig = SecurityConfig(jwt_secret_key="change-me")
    embedding: EmbeddingConfig = EmbeddingConfig()
    rag: RAGConfig = RAGConfig()
    debug: bool = False
    log_level: str = "INFO"