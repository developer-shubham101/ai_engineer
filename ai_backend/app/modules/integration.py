"""Dependency injection container for modular architecture."""

from typing import Dict, Any, Optional
import logging

from .auth.jwt_auth import JWTAuthenticator
from .auth.user_manager import SQLiteUserManager
from .auth.session_manager import SQLiteSessionManager
from .vector_db.chroma_impl import ChromaVectorStore
from .vector_db.embedding_manager import EmbeddingManager
from .llm.rag_orchestrator import RAGOrchestrator
from .core.document_manager import DocumentManager
from .core.version_manager import VersionManager

logger = logging.getLogger(__name__)


class Container:
    """Dependency injection container."""
    
    def __init__(self):
        self._instances: Dict[str, Any] = {}
        self._initialized = False
    
    def initialize(self) -> None:
        """Initialize all services."""
        if self._initialized:
            return
        
        logger.info("Initializing modular architecture container...")
        
        # Initialize core services
        self._instances["user_manager"] = SQLiteUserManager()
        self._instances["session_manager"] = SQLiteSessionManager()
        self._instances["embedding_manager"] = EmbeddingManager()
        self._instances["vector_store"] = ChromaVectorStore(self._instances["embedding_manager"])
        self._instances["version_manager"] = VersionManager()
        self._instances["document_manager"] = DocumentManager(
            self._instances["vector_store"], 
            self._instances["version_manager"],
            self._instances["embedding_manager"]
        )
        
        # Initialize auth with user manager
        authenticator = JWTAuthenticator()
        authenticator.user_manager = self._instances["user_manager"]
        self._instances["authenticator"] = authenticator
        
        # Initialize RAG orchestrator
        self._instances["rag_orchestrator"] = RAGOrchestrator(
            self._instances["vector_store"],
            self._instances["session_manager"]
        )
        
        self._initialized = True
        logger.info("Container initialized successfully")
    
    def get_user_manager(self) -> SQLiteUserManager:
        """Get user manager instance."""
        return self._instances.get("user_manager")
    
    def get_session_manager(self) -> SQLiteSessionManager:
        """Get session manager instance."""
        return self._instances.get("session_manager")
    
    def get_authenticator(self) -> JWTAuthenticator:
        """Get authenticator instance."""
        return self._instances.get("authenticator")
    
    def get_vector_store(self) -> ChromaVectorStore:
        """Get vector store instance."""
        return self._instances.get("vector_store")
    
    def get_embedding_manager(self) -> EmbeddingManager:
        """Get embedding manager instance."""
        return self._instances.get("embedding_manager")
    
    def get_document_manager(self) -> DocumentManager:
        """Get document manager instance."""
        return self._instances.get("document_manager")
    
    def get_version_manager(self) ->VersionManager:
        """Get version manager instance."""
        return self._instances.get("version_manager")
    
    def get_rag_orchestrator(self) -> RAGOrchestrator:
        """Get RAG orchestrator instance."""
        return self._instances.get("rag_orchestrator")
    
    def override_instance(self, key: str, instance: Any) -> None:
        """Override an instance (useful for testing)."""
        self._instances[key] = instance


# Global container instance
_container: Optional[Container] = None


def get_container() -> Container:
    """Get global container instance."""
    global _container
    if _container is None:
        _container = Container()
    return _container


def reset_container() -> None:
    """Reset container (useful for testing)."""
    global _container
    _container = None