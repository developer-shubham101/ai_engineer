"""Dependency injection container for modular architecture."""

from typing import Dict, Any, Optional, Union
import logging
import os

from .agents.interfaces import IAgentOrchestrator
from .auth.jwt_auth import JWTAuthenticator
from .auth.user_manager import SQLiteUserManager
from .auth.session_manager import SQLiteSessionManager
from .conversation.conversation_manager import SQLiteConversationManager
from .vector_db.chroma_impl import ChromaVectorStore
from .vector_db.faiss_vector_store import FaissVectorStore
from .vector_db.embedding_manager import EmbeddingManager
from .llm.rag_orchestrator import RAGOrchestrator
from .core.document_manager import DocumentManager
from .core.version_manager import VersionManager
from .llm.template_manager import TemplateManager
from .agents.factories import AgentOrchestratorFactory

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
        # Choose vector store based on environment variable
        vector_store_type = os.getenv("VECTOR_STORE_TYPE", "faiss").lower()
        if vector_store_type == "faiss":
            self._instances["vector_store"] = FaissVectorStore(self._instances["embedding_manager"])
            logger.info("Using FaissVectorStore")
        else:
            self._instances["vector_store"] = ChromaVectorStore(self._instances["embedding_manager"])
            logger.info("Using ChromaVectorStore")
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
        
        # Initialize RAG orchestrator with conversation manager
        self._instances["rag_orchestrator"] = RAGOrchestrator(
            self._instances["vector_store"],
            self._instances["session_manager"],
            conversation_manager=self.get_conversation_manager(),
            template_manager=self.get_template_manager()
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
    
    def get_vector_store(self) -> Union[ChromaVectorStore, FaissVectorStore]:
        """Get vector store instance."""
        return self._instances.get("vector_store")
    
    def get_embedding_manager(self) -> EmbeddingManager:
        """Get embedding manager instance."""
        return self._instances.get("embedding_manager")
    
    def get_reranker(self):
        """Get reranker instance."""
        if "reranker" not in self._instances:
            from .vector_db.reranker import CrossEncoderReranker
            self._instances["reranker"] = CrossEncoderReranker()
            logger.info("Initialized cross-encoder reranker")
        return self._instances.get("reranker")
    
    def get_bm25_index(self):
        """Get BM25 index instance."""
        if "bm25_index" not in self._instances:
            from .vector_db.bm25_index import BM25Index
            self._instances["bm25_index"] = BM25Index()
            logger.info("Initialized BM25 index")
        return self._instances.get("bm25_index")
    
    def get_document_manager(self) -> DocumentManager:
        """Get document manager instance."""
        return self._instances.get("document_manager")
    
    def get_version_manager(self) ->VersionManager:
        """Get version manager instance."""
        return self._instances.get("version_manager")
    
    def get_rag_orchestrator(self) -> RAGOrchestrator:
        """Get RAG orchestrator instance."""
        return self._instances.get("rag_orchestrator")
    
    def get_conversation_manager(self) -> SQLiteConversationManager:
        """Get conversation manager instance."""
        if "conversation_manager" not in self._instances:
            from .config.settings import settings
            db_path = settings.DATABASE_DIR / settings.CONVERSATIONS_DB_NAME
            self._instances["conversation_manager"] = SQLiteConversationManager(db_path)
            logger.info(f"Initialized conversation manager with db at {db_path}")
        return self._instances.get("conversation_manager")
    
    def get_template_manager(self) -> TemplateManager:
        """Get template manager instance."""
        if "template_manager" not in self._instances:
            self._instances["template_manager"] = TemplateManager()
            logger.info("Initialized template manager")
        return self._instances.get("template_manager")
    
    def get_agent_orchestrator(self) -> IAgentOrchestrator:
        """Get agent orchestrator instance."""
        if "agent_orchestrator" not in self._instances:
            vector_store = self.get_vector_store()
            self._instances["agent_orchestrator"] = AgentOrchestratorFactory.create_orchestrator(
                vector_store=vector_store)
            logger.info("Initialized agent orchestrator")
        return self._instances.get("agent_orchestrator")
    
    def get_metadata_generator(self):
        """Get metadata generator instance."""
        if "metadata_generator" not in self._instances:
            # Local LLM archived — use LlamaServer instead.
            # See archive/local_llm/ for the original LocalLLMProvider implementation.
            from .llm.providers.llamaserver import LlamaServerProvider
            from .core.metadata_generator import LLMMetadataGenerator
            llm_provider = LlamaServerProvider(configs={})
            self._instances["metadata_generator"] = LLMMetadataGenerator(llm_provider)
            logger.info("Initialized metadata generator with LlamaServer provider")
        return self._instances.get("metadata_generator")

    
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