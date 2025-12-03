"""Vector database interfaces."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class IVectorStore(ABC):
    """Interface for vector database operations."""

    @abstractmethod
    async def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add document to vector store."""
        pass

    @abstractmethod
    async def search_documents(self, query: str, top_k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> \
    List[Dict[str, Any]]:
        """Search for similar documents."""
        pass

    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete document from vector store."""
        pass

    @abstractmethod
    async def update_document(self, document_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update document in vector store."""
        pass

    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        pass


class IEmbeddingManager(ABC):
    """Interface for embedding management."""

    @abstractmethod
    async def encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts to embeddings."""
        pass

    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get embedding model information."""
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension."""
        pass
