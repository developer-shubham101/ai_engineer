"""Vector database interfaces."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


class IEmbeddingProvider(ABC):
    """Interface for embedding providers."""
    
    @abstractmethod
    async def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        pass
    
    @abstractmethod
    async def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding."""
        pass
    
    @abstractmethod
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get model name."""
        pass


class IVectorDatabase(ABC):
    """Interface for vector database operations."""
    
    @abstractmethod
    async def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> List[str]:
        """Add documents with embeddings to the database."""
        pass
    
    @abstractmethod
    async def search(self, query_embedding: np.ndarray, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        pass
    
    @abstractmethod
    async def update_document(self, document_id: str, document: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> bool:
        """Update document."""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete document."""
        pass
    
    @abstractmethod
    async def list_documents(self, filter_metadata: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List documents with optional filtering."""
        pass
    
    @abstractmethod
    async def count_documents(self, filter_metadata: Optional[Dict[str, Any]] = None) -> int:
        """Count documents with optional filtering."""
        pass
    
    @abstractmethod
    async def clear_collection(self) -> bool:
        """Clear all documents from collection."""
        pass


class IDocumentProcessor(ABC):
    """Interface for document processing."""
    
    @abstractmethod
    async def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document into chunks."""
        pass
    
    @abstractmethod
    async def process_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process document and return chunks with metadata."""
        pass
    
    @abstractmethod
    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate document metadata."""
        pass


class IVectorStore(ABC):
    """High-level interface combining vector database and embedding provider."""
    
    @abstractmethod
    async def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add single document."""
        pass
    
    @abstractmethod
    async def add_documents(self, documents: List[Tuple[str, Dict[str, Any]]]) -> List[str]:
        """Add multiple documents."""
        pass
    
    @abstractmethod
    async def search_documents(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search documents by text query."""
        pass
    
    @abstractmethod
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        pass
    
    @abstractmethod
    async def update_document(self, document_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update document."""
        pass
    
    @abstractmethod
    async def delete_document(self, document_id: str) -> bool:
        """Delete document."""
        pass
    
    @abstractmethod
    async def list_documents(self, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List documents."""
        pass