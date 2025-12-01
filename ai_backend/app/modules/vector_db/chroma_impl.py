"""ChromaDB implementation of vector database."""

import chromadb
from chromadb.config import Settings as ChromaSettings
import numpy as np
from typing import List, Dict, Any, Optional
import uuid
import logging

from .interfaces import IVectorDatabase, IVectorStore, IDocumentProcessor
from .embedding_manager import EmbeddingManager
from ..config.settings import settings

logger = logging.getLogger(__name__)


class ChromaVectorDatabase(IVectorDatabase):
    """ChromaDB implementation of vector database."""
    
    def __init__(self, collection_name: str = "rag_documents"):
        self.collection_name = collection_name
        self.client = None
        self.collection = None
        self._init_client()
    
    def _init_client(self):
        """Initialize ChromaDB client."""
        try:
            self.client = chromadb.PersistentClient(
                path=str(settings.CHROMA_STORAGE_DIR),
                settings=ChromaSettings(anonymized_telemetry=False)
            )
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    async def add_documents(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> List[str]:
        """Add documents with embeddings to the database."""
        try:
            ids = [doc.get("id", str(uuid.uuid4())) for doc in documents]
            texts = [doc["text"] for doc in documents]
            metadatas = [doc.get("metadata", {}) for doc in documents]
            
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings.tolist(),
                metadatas=metadatas
            )
            
            logger.info(f"Added {len(documents)} documents to ChromaDB")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise
    
    async def search(self, query_embedding: np.ndarray, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            where_clause = filter_metadata if filter_metadata else None
            
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k,
                where=where_clause
            )
            
            documents = []
            if results["ids"] and results["ids"][0]:
                for i, doc_id in enumerate(results["ids"][0]):
                    documents.append({
                        "id": doc_id,
                        "text": results["documents"][0][i],
                        "metadata": results["metadatas"][0][i],
                        "distance": results["distances"][0][i]
                    })
            
            logger.debug(f"Found {len(documents)} similar documents")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to search documents in ChromaDB: {e}")
            raise
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        try:
            results = self.collection.get(ids=[document_id])
            
            if results["ids"]:
                return {
                    "id": results["ids"][0],
                    "text": results["documents"][0],
                    "metadata": results["metadatas"][0]
                }
            return None
            
        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    async def update_document(self, document_id: str, document: Dict[str, Any], embedding: Optional[np.ndarray] = None) -> bool:
        """Update document."""
        try:
            update_data = {
                "ids": [document_id],
                "documents": [document["text"]],
                "metadatas": [document.get("metadata", {})]
            }
            
            if embedding is not None:
                update_data["embeddings"] = [embedding.tolist()]
            
            self.collection.update(**update_data)
            logger.info(f"Updated document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document."""
        try:
            self.collection.delete(ids=[document_id])
            logger.info(f"Deleted document: {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False
    
    async def list_documents(self, filter_metadata: Optional[Dict[str, Any]] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List documents with optional filtering."""
        try:
            where_clause = filter_metadata if filter_metadata else None
            
            results = self.collection.get(
                where=where_clause,
                limit=limit
            )
            
            documents = []
            if results["ids"]:
                for i, doc_id in enumerate(results["ids"]):
                    documents.append({
                        "id": doc_id,
                        "text": results["documents"][i],
                        "metadata": results["metadatas"][i]
                    })
            
            return documents
            
        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []
    
    async def count_documents(self, filter_metadata: Optional[Dict[str, Any]] = None) -> int:
        """Count documents with optional filtering."""
        try:
            documents = await self.list_documents(filter_metadata)
            return len(documents)
            
        except Exception as e:
            logger.error(f"Failed to count documents: {e}")
            return 0
    
    async def clear_collection(self) -> bool:
        """Clear all documents from collection."""
        try:
            # Delete the collection and recreate it
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Cleared collection: {self.collection_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False


class SimpleDocumentProcessor(IDocumentProcessor):
    """Simple document processor implementation."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    async def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split document into chunks."""
        chunks = []
        
        # Simple character-based chunking
        start = 0
        chunk_id = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end]
            
            # Try to break at word boundary
            if end < len(text):
                last_space = chunk_text.rfind(' ')
                if last_space > self.chunk_size // 2:
                    chunk_text = chunk_text[:last_space]
                    end = start + last_space
            
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = chunk_id
            chunk_metadata["start_pos"] = start
            chunk_metadata["end_pos"] = end
            
            chunks.append({
                "text": chunk_text.strip(),
                "metadata": chunk_metadata
            })
            
            start = end - self.chunk_overlap
            chunk_id += 1
        
        return chunks
    
    async def process_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process document and return chunks with metadata."""
        if not self.validate_metadata(metadata):
            raise ValueError("Invalid document metadata")
        
        return await self.chunk_document(text, metadata)
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> bool:
        """Validate document metadata."""
        required_fields = ["source"]
        return all(field in metadata for field in required_fields)


class ChromaVectorStore(IVectorStore):
    """High-level ChromaDB vector store implementation."""
    
    def __init__(self, collection_name: str = "rag_documents"):
        self.vector_db = ChromaVectorDatabase(collection_name)
        self.embedding_provider = EmbeddingManager.get_provider()
        self.document_processor = SimpleDocumentProcessor()
    
    async def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add single document."""
        document_id = str(uuid.uuid4())
        
        # Process document into chunks
        chunks = await self.document_processor.process_document(text, metadata)
        
        # Generate embeddings
        chunk_texts = [chunk["text"] for chunk in chunks]
        embeddings = await self.embedding_provider.encode(chunk_texts)
        
        # Add document ID to each chunk
        for i, chunk in enumerate(chunks):
            chunk["id"] = f"{document_id}_chunk_{i}"
            chunk["metadata"]["document_id"] = document_id
        
        # Store in vector database
        await self.vector_db.add_documents(chunks, embeddings)
        
        logger.info(f"Added document {document_id} with {len(chunks)} chunks")
        return document_id
    
    async def add_documents(self, documents: List[tuple]) -> List[str]:
        """Add multiple documents."""
        document_ids = []
        
        for text, metadata in documents:
            doc_id = await self.add_document(text, metadata)
            document_ids.append(doc_id)
        
        return document_ids
    
    async def search_documents(self, query: str, top_k: int = 5, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search documents by text query."""
        # Generate query embedding
        query_embedding = await self.embedding_provider.encode_single(query)
        
        # Search in vector database
        results = await self.vector_db.search(query_embedding, top_k, filter_metadata)
        
        return results
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by ID."""
        return await self.vector_db.get_document(document_id)
    
    async def update_document(self, document_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update document."""
        # Generate new embedding
        embedding = await self.embedding_provider.encode_single(text)
        
        document = {
            "text": text,
            "metadata": metadata
        }
        
        return await self.vector_db.update_document(document_id, document, embedding)
    
    async def delete_document(self, document_id: str) -> bool:
        """Delete document."""
        return await self.vector_db.delete_document(document_id)
    
    async def list_documents(self, filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List documents."""
        return await self.vector_db.list_documents(filter_metadata)