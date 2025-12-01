"""Module integration and dependency injection."""

from typing import Dict, Any, Optional
import logging

# Import all module interfaces and implementations
from .api.models import QueryRequest, QueryResponse, AuthRequest, AuthResponse
from .api.handlers import RAGHandler, AuthHandler
from .api.validators import QueryValidator, DocumentValidator, UserValidator

from .config.settings import settings
from .config.constants import *

from .auth.interfaces import IAuthenticator, IUserManager, ISessionManager, IRBACManager
from .auth.jwt_auth import JWTAuthenticator
from .auth.user_manager import SQLiteUserManager
from .auth.session_manager import SQLiteSessionManager
from .auth.rbac import RBACManager

from .vector_db.interfaces import IVectorStore, IEmbeddingProvider
from .vector_db.chroma_impl import ChromaVectorStore
from .vector_db.embedding_manager import EmbeddingManager

from .llm.interfaces import ILLMProvider, IRAGOrchestrator
from .llm.providers import ProviderFactory
from .llm.rag_orchestrator import MultiProviderRAGOrchestrator

from .core.document_manager import DocumentManager
from .core.version_manager import VersionManager
from .core.profile_analyzer import ProfileAnalyzer

logger = logging.getLogger(__name__)


class ModuleContainer:
    """Dependency injection container for all modules."""
    
    def __init__(self):
        self._instances: Dict[str, Any] = {}
        self._initialized = False
    
    def initialize(self):
        """Initialize all modules and their dependencies."""
        if self._initialized:
            return
        
        logger.info("Initializing modular architecture...")
        
        # 1. Initialize configuration
        self._instances["settings"] = settings
        
        # 2. Initialize authentication modules
        self._instances["user_manager"] = SQLiteUserManager()
        self._instances["session_manager"] = SQLiteSessionManager()
        self._instances["rbac_manager"] = RBACManager()
        
        # Initialize authenticator with user manager
        authenticator = JWTAuthenticator()
        authenticator.user_manager = self._instances["user_manager"]
        self._instances["authenticator"] = authenticator
        
        # 3. Initialize vector database modules
        self._instances["embedding_provider"] = EmbeddingManager.get_provider()
        self._instances["vector_store"] = ChromaVectorStore()
        
        # 4. Initialize LLM modules
        self._instances["rag_orchestrator"] = MultiProviderRAGOrchestrator(
            vector_store=self._instances["vector_store"],
            rbac_manager=self._instances["rbac_manager"],
            session_manager=self._instances["session_manager"]
        )
        
        # Register LLM providers
        orchestrator = self._instances["rag_orchestrator"]
        
        # Register available providers
        for provider_name in ["mock", "local"]:  # Add more as needed
            try:
                provider = ProviderFactory.create_provider(provider_name)
                if provider.is_available():
                    orchestrator.register_provider(provider_name, provider)
                    logger.info(f"Registered LLM provider: {provider_name}")
            except Exception as e:
                logger.warning(f"Failed to register provider {provider_name}: {e}")
        
        # 5. Initialize core services
        self._instances["document_manager"] = DocumentManager(
            vector_store=self._instances["vector_store"],
            rbac_manager=self._instances["rbac_manager"]
        )
        
        self._instances["version_manager"] = VersionManager()
        
        self._instances["profile_analyzer"] = ProfileAnalyzer(
            user_manager=self._instances["user_manager"],
            session_manager=self._instances["session_manager"]
        )
        
        # 6. Initialize API handlers
        self._instances["rag_handler"] = RAGHandler()
        self._instances["auth_handler"] = AuthHandler()
        
        # 7. Initialize validators
        self._instances["query_validator"] = QueryValidator()
        self._instances["document_validator"] = DocumentValidator()
        self._instances["user_validator"] = UserValidator()
        
        self._initialized = True
        logger.info("Modular architecture initialized successfully")
    
    def get(self, service_name: str) -> Any:
        """Get service instance by name."""
        if not self._initialized:
            self.initialize()
        
        if service_name not in self._instances:
            raise ValueError(f"Service not found: {service_name}")
        
        return self._instances[service_name]
    
    def get_authenticator(self) -> IAuthenticator:
        """Get authenticator instance."""
        return self.get("authenticator")
    
    def get_user_manager(self) -> IUserManager:
        """Get user manager instance."""
        return self.get("user_manager")
    
    def get_session_manager(self) -> ISessionManager:
        """Get session manager instance."""
        return self.get("session_manager")
    
    def get_rbac_manager(self) -> IRBACManager:
        """Get RBAC manager instance."""
        return self.get("rbac_manager")
    
    def get_vector_store(self) -> IVectorStore:
        """Get vector store instance."""
        return self.get("vector_store")
    
    def get_rag_orchestrator(self) -> IRAGOrchestrator:
        """Get RAG orchestrator instance."""
        return self.get("rag_orchestrator")
    
    def get_document_manager(self) -> DocumentManager:
        """Get document manager instance."""
        return self.get("document_manager")
    
    def get_version_manager(self) -> VersionManager:
        """Get version manager instance."""
        return self.get("version_manager")
    
    def get_profile_analyzer(self) -> ProfileAnalyzer:
        """Get profile analyzer instance."""
        return self.get("profile_analyzer")
    
    def reset(self):
        """Reset container (useful for testing)."""
        self._instances.clear()
        self._initialized = False
        EmbeddingManager.reset()  # Reset singleton


# Global container instance
container = ModuleContainer()


def get_container() -> ModuleContainer:
    """Get global container instance."""
    return container