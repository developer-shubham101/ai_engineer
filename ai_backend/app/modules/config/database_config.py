"""Database configuration for the application."""

from pathlib import Path
from typing import Dict, Any


class DatabaseConfig:
    """Centralized database configuration."""
    
    # Database file names
    USERS_DB = "users.db"
    SESSIONS_DB = "support_sessions.db"
    DOCUMENT_VERSIONS_DB = "document_versions.db"
    
    # Collection names
    DEFAULT_COLLECTION_NAME = "local_manual_rag"
    DOCUMENTS_COLLECTION = "documents"
    
    # Database settings
    SQLITE_TIMEOUT = 30.0  # seconds
    SQLITE_CHECK_SAME_THREAD = False
    
    # ChromaDB settings
    CHROMA_ANONYMIZED_TELEMETRY = False
    CHROMA_ALLOW_RESET = True
    
    @classmethod
    def get_db_path(cls, db_name: str, base_dir: Path) -> Path:
        """Get full path for a database file."""
        return base_dir / db_name
    
    @classmethod
    def to_dict(cls) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "users_db": cls.USERS_DB,
            "sessions_db": cls.SESSIONS_DB,
            "document_versions_db": cls.DOCUMENT_VERSIONS_DB,
            "default_collection_name": cls.DEFAULT_COLLECTION_NAME,
            "documents_collection": cls.DOCUMENTS_COLLECTION,
            "sqlite_timeout": cls.SQLITE_TIMEOUT,
            "chroma_anonymized_telemetry": cls.CHROMA_ANONYMIZED_TELEMETRY,
        }


# Global database config instance
db_config = DatabaseConfig()
