"""FAISS implementation for vector storage."""

import logging
import uuid
import faiss
import numpy as np
import pickle
from typing import Any, Dict, List, Optional

from .interfaces import IVectorStore, IEmbeddingManager

logger = logging.getLogger(__name__)

class FaissVectorStore(IVectorStore):
    """FAISS implementation of vector store."""

    def __init__(self, embedding_manager: IEmbeddingManager, file_path: str = "faiss_index.pkl"):
        self.embedding_manager = embedding_manager
        self.file_path = file_path
        self.index = None
        self.documents = {}  # Stores text and metadata
        self._initialized = False
        self.dimension = self.embedding_manager.get_embedding_dimension()
        self._load_index()

    def _load_index(self):
        try:
            with open(self.file_path, "rb") as f:
                data = pickle.load(f)
                self.index = faiss.deserialize_index(data["index"])
                self.documents = data["documents"]
            self._initialized = True
            logger.info(f"FAISS index loaded from {self.file_path}")
        except (FileNotFoundError, EOFError):
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = {}
            self._initialized = True
            logger.info("No FAISS index found. Initialized a new one.")

    def _save_index(self):
        with open(self.file_path, "wb") as f:
            data = {
                "index": faiss.serialize_index(self.index),
                "documents": self.documents
            }
            pickle.dump(data, f)
        logger.info(f"FAISS index saved to {self.file_path}")

    async def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add document to vector store."""
        try:
            doc_id = str(uuid.uuid4())
            embedding = await self.embedding_manager.encode([text])
            embedding_np = np.array(embedding).astype('float32')
            
            self.index.add(embedding_np)
            doc_index = self.index.ntotal - 1
            
            self.documents[doc_index] = {"id": doc_id, "text": text, "metadata": metadata}
            
            self._save_index()
            logger.info(f"Added document: {doc_id} at index {doc_index}")
            return doc_id
        except Exception as e:
            logger.exception("Failed to add document: %s", e)
            raise

    async def search_documents(self, query: str, top_k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            query_embedding = await self.embedding_manager.encode([query])
            query_embedding_np = np.array(query_embedding).astype('float32')
            
            distances, indices = self.index.search(query_embedding_np, top_k)
            
            results = []
            for i, idx in enumerate(indices[0]):
                if idx in self.documents:
                    doc_info = self.documents[idx]
                    
                    # Basic metadata filtering
                    if metadata_filter:
                        metadata = doc_info.get("metadata", {})
                        match = all(metadata.get(k) == v for k, v in metadata_filter.items())
                        if not match:
                            continue
                            
                    results.append({
                        "id": doc_info["id"],
                        "text": doc_info["text"],
                        "metadata": doc_info["metadata"],
                        "distance": float(distances[0][i])
                    })
            return results
        except Exception as e:
            logger.exception("Failed to search documents: %s", e)
            return []

    async def delete_document(self, document_id: str) -> bool:
        """Delete document from vector store.
        Note: FAISS does not support direct deletion by ID.
        This is a placeholder and does not fully remove the vector.
        """
        logger.warning("FAISS does not support efficient deletion. Document will be marked as deleted.")
        
        # Find the index corresponding to the document_id
        for index, doc in self.documents.items():
            if doc['id'] == document_id:
                # Mark as deleted
                del self.documents[index]
                self._save_index()
                logger.info(f"Document {document_id} marked as deleted.")
                return True
        return False

    async def update_document(self, document_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update document in vector store.
        Note: This is a simple implementation that deletes the old entry and adds a new one.
        """
        if await self.delete_document(document_id):
            await self.add_document(text, metadata)
            logger.info(f"Document {document_id} updated.")
            return True
        return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        return {
            "name": "faiss_index",
            "document_count": self.index.ntotal,
            "embedding_dimension": self.dimension
        }

    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by its ID."""
        for doc in self.documents.values():
            if doc['id'] == document_id:
                return doc
        return None

    # =========================================================================
    # Batch Operations
    # =========================================================================

    def add_documents_to_collection(self,
                                    documents: List[str],
                                    metadatas: List[Dict[str, Any]],
                                    ids: List[str],
                                    embeddings: Optional[List[List[float]]] = None) -> None:
        """Add multiple documents to the collection with optional pre-computed embeddings."""
        try:
            if embeddings is None:
                # This is a sync method but we need async encode - log warning
                logger.warning("add_documents_to_collection called without embeddings. This requires async operation.")
                raise ValueError("Embeddings must be pre-computed for batch operations in FAISS")
            
            embeddings_np = np.array(embeddings).astype('float32')
            
            # Add all embeddings to the index
            start_index = self.index.ntotal
            self.index.add(embeddings_np)
            
            # Store document metadata
            for i, (doc_id, text, metadata) in enumerate(zip(ids, documents, metadatas)):
                doc_index = start_index + i
                self.documents[doc_index] = {
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata
                }
            
            self._save_index()
            logger.info(f"Added {len(documents)} documents to FAISS collection")
            
        except Exception as e:
            logger.exception("Failed to add documents to collection: %s", e)
            raise

    def get_documents_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """Get multiple documents by their IDs."""
        try:
            result_ids = []
            result_documents = []
            result_metadatas = []
            
            for doc_index, doc_info in self.documents.items():
                if doc_info["id"] in ids:
                    result_ids.append(doc_info["id"])
                    result_documents.append(doc_info["text"])
                    result_metadatas.append(doc_info["metadata"])
            
            return {
                "ids": result_ids,
                "documents": result_documents,
                "metadatas": result_metadatas
            }
        except Exception as e:
            logger.exception("Failed to get documents by IDs: %s", e)
            raise

    def delete_ids(self, ids: List[str]) -> None:
        """Delete multiple documents by their IDs.
        Note: FAISS does not support efficient deletion. Documents will be marked as deleted.
        """
        try:
            deleted_count = 0
            indices_to_delete = []
            
            for doc_index, doc_info in self.documents.items():
                if doc_info["id"] in ids:
                    indices_to_delete.append(doc_index)
            
            for idx in indices_to_delete:
                del self.documents[idx]
                deleted_count += 1
            
            if deleted_count > 0:
                self._save_index()
            
            logger.info(f"Deleted {deleted_count} documents from FAISS collection")
            
        except Exception as e:
            logger.exception("Failed to delete documents by IDs: %s", e)
            raise

    def update_metadatas(self, ids: List[str], metadata: Dict[str, Any]) -> bool:
        """Update metadata for multiple documents."""
        try:
            updated_count = 0
            
            for doc_index, doc_info in self.documents.items():
                if doc_info["id"] in ids:
                    # Update metadata while preserving existing keys
                    doc_info["metadata"].update(metadata)
                    updated_count += 1
            
            if updated_count > 0:
                self._save_index()
                logger.info(f"Updated metadata for {updated_count} documents")
                return True
            
            return False
            
        except Exception as e:
            logger.exception("Failed to update metadatas: %s", e)
            return False

    # =========================================================================
    # Query Operations
    # =========================================================================

    def query_collection(self,
                         query_embeddings: Optional[List[List[float]]] = None,
                         query_texts: Optional[List[str]] = None,
                         n_results: int = 3) -> Dict[str, Any]:
        """Query the collection using embeddings or text.
        Note: This is a synchronous method but query_texts requires async encoding.
        """
        try:
            if query_embeddings is None and query_texts is None:
                raise ValueError("Either query_embeddings or query_texts must be provided")
            
            if query_texts is not None:
                logger.warning("query_collection with query_texts requires async operation")
                raise ValueError("query_texts not supported in sync method. Use query_embeddings or search_documents instead")
            
            # Use provided embeddings
            query_embeddings_np = np.array(query_embeddings).astype('float32')
            distances, indices = self.index.search(query_embeddings_np, n_results)
            
            # Format results similar to ChromaDB
            result_ids = []
            result_documents = []
            result_metadatas = []
            result_distances = []
            
            for query_idx in range(len(query_embeddings)):
                query_result_ids = []
                query_result_docs = []
                query_result_metas = []
                query_result_dists = []
                
                for i, idx in enumerate(indices[query_idx]):
                    if idx in self.documents and idx != -1:  # -1 indicates no result
                        doc_info = self.documents[idx]
                        query_result_ids.append(doc_info["id"])
                        query_result_docs.append(doc_info["text"])
                        query_result_metas.append(doc_info["metadata"])
                        query_result_dists.append(float(distances[query_idx][i]))
                
                result_ids.append(query_result_ids)
                result_documents.append(query_result_docs)
                result_metadatas.append(query_result_metas)
                result_distances.append(query_result_dists)
            
            return {
                "ids": result_ids,
                "documents": result_documents,
                "metadatas": result_metadatas,
                "distances": result_distances
            }
            
        except Exception as e:
            logger.exception("Failed to query collection: %s", e)
            raise

    # =========================================================================
    # Collection Management
    # =========================================================================

    def get_collection_data(self) -> Dict[str, Any]:
        """Get all data from the collection."""
        try:
            all_ids = []
            all_documents = []
            all_metadatas = []
            
            for doc_info in self.documents.values():
                all_ids.append(doc_info["id"])
                all_documents.append(doc_info["text"])
                all_metadatas.append(doc_info["metadata"])
            
            return {
                "ids": all_ids,
                "documents": all_documents,
                "metadatas": all_metadatas
            }
            
        except Exception as e:
            logger.exception("Failed to get collection data: %s", e)
            raise

    def delete_all_documents(self) -> None:
        """Delete all documents from the collection."""
        try:
            # Reinitialize the index
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = {}
            self._save_index()
            logger.info("Deleted all documents from FAISS collection")
            
        except Exception as e:
            logger.exception("Failed to delete all documents: %s", e)
            raise

    def delete_collection_by_name(self) -> None:
        """Delete the entire collection.
        For FAISS, this is equivalent to deleting all documents and resetting the index.
        """
        try:
            import os
            
            # Delete the index file if it exists
            if os.path.exists(self.file_path):
                os.remove(self.file_path)
                logger.info(f"Deleted FAISS index file: {self.file_path}")
            
            # Reinitialize
            self.index = faiss.IndexFlatL2(self.dimension)
            self.documents = {}
            self._initialized = True
            
            logger.info("Deleted FAISS collection")
            
        except Exception as e:
            logger.exception("Failed to delete collection: %s", e)
            raise
