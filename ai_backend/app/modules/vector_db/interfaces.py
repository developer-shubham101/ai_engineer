"""Vector database interfaces."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class IVectorStore(ABC):
    """Interface for vector database operations."""

    # =========================================================================
    # Core CRUD Operations (Single Document)
    # =========================================================================

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

    @abstractmethod
    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by its ID."""
        pass

    # =========================================================================
    # Batch Operations
    # =========================================================================

    @abstractmethod
    def add_documents_to_collection(self,
                                    documents: List[str],
                                    metadatas: List[Dict[str, Any]],
                                    ids: List[str],
                                    embeddings: Optional[List[List[float]]] = None) -> None:
        """Add multiple documents to the collection with optional pre-computed embeddings."""
        pass

    @abstractmethod
    def get_documents_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """Get multiple documents by their IDs."""
        pass

    @abstractmethod
    def delete_ids(self, ids: List[str]) -> None:
        """Delete multiple documents by their IDs."""
        pass

    @abstractmethod
    def update_metadatas(self, ids: List[str], metadata: Dict[str, Any]) -> bool:
        """Update metadata for multiple documents."""
        pass

    # =========================================================================
    # Query Operations
    # =========================================================================

    @abstractmethod
    def query_collection(self,
                         query_embeddings: Optional[List[List[float]]] = None,
                         query_texts: Optional[List[str]] = None,
                         n_results: int = 3) -> Dict[str, Any]:
        """Query the collection using embeddings or text."""
        pass

    # =========================================================================
    # Collection Management
    # =========================================================================

    @abstractmethod
    def get_collection_data(self) -> Dict[str, Any]:
        """Get all data from the collection."""
        pass

    @abstractmethod
    def delete_all_documents(self) -> None:
        """Delete all documents from the collection."""
        pass

    @abstractmethod
    def delete_collection_by_name(self) -> None:
        """Delete the entire collection."""
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
