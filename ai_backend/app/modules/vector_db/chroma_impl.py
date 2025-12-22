"""ChromaDB implementation for vector storage."""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple
from app.modules.config.settings import settings

try:
    # Attempt to import chromadb and its configuration settings
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None  # Will raise ImportError at runtime if used without installation

from .interfaces import IVectorStore, IEmbeddingManager


DEFAULT_PERSIST_DIR = settings.DEFAULT_PERSIST_DIR
DEFAULT_COLLECTION_NAME = settings.DEFAULT_COLLECTION_NAME


logger = logging.getLogger(__name__)


class ChromaVectorStore(IVectorStore):
    """
    ChromaDB implementation of vector store, incorporating all utility logic.
    """

    def __init__(self, embedding_manager: IEmbeddingManager,
                 persist_directory: Optional[str] = None,
                 collection_name: Optional[str] = None):

        if chromadb is None:
            raise ImportError("chromadb is not installed. Install chromadb to use ChromaVectorStore.")

        self.embedding_manager = embedding_manager
        self.persist_directory = persist_directory or str(DEFAULT_PERSIST_DIR)
        self.collection_name = collection_name or DEFAULT_COLLECTION_NAME

        # Internal ChromaDB objects
        self._client: Any = None
        self._collection: Any = None
        self._initialized = False

        # Initialize the client and collection immediately
        self.ensure_chroma_client()


    # =========================================================================
    # Utility Functions (Moved from chroma_utils.py / ChromaClientManager)
    # These encapsulate the raw ChromaDB API calls
    # =========================================================================

    def ensure_chroma_client(self) -> Tuple[Any, Any]:
        """
        Initialize (or return cached) chroma client and collection.
        (Original function name)
        """
        if self._initialized:
            return self._client, self._collection

        # Try new API first, fallback to older APIs
        try:
            from chromadb.config import Settings as _Settings
            self._client = chromadb.Client(_Settings(chroma_db_impl="duckdb+parquet", persist_directory=self.persist_directory))
        except Exception:
            try:
                self._client = chromadb.PersistentClient(path=str(self.persist_directory))
            except Exception as e:
                logger.exception("Failed to initialize chroma client: %s", e)
                raise

        try:
            self._collection = self._client.get_or_create_collection(name=self.collection_name)
        except Exception:
            # fallback: try without options
            self._collection = self._client.get_or_create_collection(name=self.collection_name)

        self._initialized = True
        logger.info(f"Chroma client initialized for collection: {self.collection_name}")
        return self._client, self._collection

    def add_documents_to_collection(self,
                                    documents: List[str],
                                    metadatas: List[Dict[str, Any]],
                                    ids: List[str],
                                    embeddings: Optional[List[List[float]]] = None) -> None:
        """
        Add documents to the managed collection. (Original function name)
        """
        collection = self._collection

        logger.info("Adding %d documents to collection", len(documents))

        # Clean metadatas to ensure no None values
        cleaned_metadatas = []
        for metadata in metadatas:
            cleaned_metadata = {k: (v if v is not None else "") for k, v in metadata.items()}
            cleaned_metadatas.append(cleaned_metadata)
        metadatas = cleaned_metadatas

        try:
            if embeddings is not None:
                collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
            else:
                collection.add(documents=documents, metadatas=metadatas, ids=ids)
            logger.info("Added %d documents to collection", len(documents))
            return
        except Exception as e:
            logger.exception("collection.add failed: %s", e)
            raise


    def query_collection(self,
                         query_embeddings: Optional[List[List[float]]] = None,
                         query_texts: Optional[List[str]] = None,
                         n_results: int = 3) -> Dict[str, Any]:
        """
        Query the managed collection by embeddings or text. (Original function name)
        """
        collection = self._collection

        try:
            if query_embeddings is not None:
                return collection.query(query_embeddings=query_embeddings, n_results=n_results)
            if query_texts is not None:
                return collection.query(query_texts=query_texts, n_results=n_results)
            raise ValueError("Either query_embeddings or query_texts must be provided")
        except Exception as e:
            logger.exception("query_collection failed: %s", e)
            raise


    def get_collection_data(self) -> Dict[str, Any]:
        """
        Return a dict-like snapshot of the collection. (Original function name)
        """
        collection = self._collection

        try:
            data = collection.get()
            if isinstance(data, dict):
                return data
            # Handle object-style response for compatibility
            result = {}
            if hasattr(data, "ids"): result["ids"] = data.ids
            if hasattr(data, "documents"): result["documents"] = data.documents
            if hasattr(data, "metadatas"): result["metadatas"] = data.metadatas
            return result
        except Exception as e:
            logger.exception("get_collection_data failed: %s", e)
            raise


    def get_documents_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Get documents/metadatas for the provided ids. (Original function name)
        """
        collection = self._collection

        try:
            return collection.get(ids=ids)
        except Exception as e:
            logger.exception("get_documents_by_ids failed: %s", e)
            raise


    def update_metadatas(self, ids: List[str], metadata: Dict[str, Any]) -> bool:
        """
        Update metadata for a list of ids via native update. (Original function name)
        """
        collection = self._collection

        try:
            per_id_metas = [metadata.copy() for _ in ids]
            collection.update(ids=ids, metadatas=per_id_metas)
            logger.info("Updated metadata for %d ids via collection.update()", len(ids))
            return True
        except Exception:
            logger.debug("collection.update() failed, or documents are missing")
            return False


    def delete_ids(self, ids: List[str]) -> None:
        """
        Delete given ids from the collection. (Original function name)
        """
        collection = self._collection

        try:
            collection.delete(ids=ids)
            logger.info("Deleted %d ids from collection", len(ids))
        except Exception:
            logger.exception("collection.delete(ids=...) failed")
            raise


    def delete_collection_by_name(self) -> None:
        """
        Delete the collection entirely using the client. (Original function name)
        """
        client = self._client

        try:
            client.delete_collection(name=self.collection_name)
            self._initialized = False # Force re-initialization on next use
            logger.info("Deleted collection %s via client.delete_collection()", self.collection_name)
        except Exception:
            logger.exception("client.delete_collection failed for %s", self.collection_name)
            raise


    def delete_all_documents(self) -> None:
        """
        Attempt to remove all documents from the collection. (Original function name)
        """
        collection = self._collection

        try:
            collection.delete()
            logger.info("Cleared collection using collection.delete()")
            return
        except Exception:
            logger.debug("collection.delete() not supported or failed; attempting alternatives")

        # try to list ids and delete
        try:
            coll_data = collection.get()
            if isinstance(coll_data, dict) and coll_data.get("ids"):
                all_ids = coll_data.get("ids", [])
                if all_ids and isinstance(all_ids, list) and isinstance(all_ids[0], list):
                    # Flatten list of lists if needed (chromadb version compatibility)
                    flat_ids = [item for sublist in all_ids for item in sublist]
                else:
                    flat_ids = all_ids

                if flat_ids:
                    collection.delete(ids=flat_ids)
                    logger.info("Cleared %d ids via delete(ids=...)", len(flat_ids))
                    return
        except Exception:
            logger.debug("Unable to list ids via collection.get()")

        # last resort: delete and recreate collection
        try:
            self.delete_collection_by_name()
            self.ensure_chroma_client() # Recreate
            logger.info("Deleted and recreated collection %s", self.collection_name)
            return
        except Exception:
            logger.exception("Failed to delete and recreate collection %s via client", self.collection_name)

        logger.warning("Unable to clear collection using available APIs; collection may still contain documents.")


    # =========================================================================
    # IVectorStore Interface Implementation (Using Utility Functions)
    # =========================================================================

    async def add_document(self, text: str, metadata: Dict[str, Any]) -> str:
        """Add document to vector store."""
        try:
            doc_id = str(uuid.uuid4())

            # Generate embedding
            embeddings = await self.embedding_manager.encode([text])

            # Use the utility method
            self.add_documents_to_collection(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id],
                embeddings=embeddings
            )

            logger.info(f"Added document: {doc_id}")
            return doc_id

        except Exception as e:
            logger.exception("Failed to add document: %s", e)
            raise

    async def search_documents(self, query: str, top_k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        try:
            # Generate query embedding
            query_embeddings = await self.embedding_manager.encode([query])

            # Use the utility method
            # NOTE: Ignoring metadata_filter for now, as query_collection only takes embeddings/text
            results = self.query_collection(
                query_embeddings=query_embeddings,
                n_results=top_k
            )

            # Convert raw ChromaDB results format to IVectorStore format
            formatted_results = []
            if results.get("ids") and results.get("documents"):
                # Handle potential list of lists structure from ChromaDB results
                ids = results["ids"][0]
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                distances = results["distances"][0]

                for i in range(len(ids)):
                    doc_id = ids[i]
                    doc_text = docs[i]
                    doc_meta = metas[i] or {}
                    doc_distance = distances[i]

                    # NOTE: A proper implementation would pass metadata_filter to query_collection
                    # or apply filtering here if query_collection cannot.

                    formatted_results.append({
                        "id": doc_id,
                        "text": doc_text,
                        "metadata": doc_meta,
                        "distance": doc_distance
                    })

            # Results are usually sorted by distance (ascending) by ChromaDB
            return formatted_results

        except Exception as e:
            logger.exception("Failed to search documents: %s", e)
            return []

    async def delete_document(self, document_id: str) -> bool:
        """Delete document from vector store."""
        try:
            # Use the utility method
            self.delete_ids(ids=[document_id])
            logger.info(f"Deleted document: {document_id}")
            return True
        except Exception as e:
            logger.exception("Failed to delete document: %s", e)
            return False

    async def update_document(self, document_id: str, text: str, metadata: Dict[str, Any]) -> bool:
        """Update document in vector store."""
        try:
            # ChromaDB uses 'add' as an upsert (update if ID exists, insert if new).

            # Generate new embedding
            embeddings = await self.embedding_manager.encode([text])

            # Use the utility method
            self.add_documents_to_collection(
                documents=[text],
                metadatas=[metadata],
                ids=[document_id],
                embeddings=embeddings
            )

            logger.info(f"Updated document: {document_id}")
            return True
        except Exception as e:
            logger.exception("Failed to update document: %s", e)
            return False

    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection."""
        # Use the utility method
        try:
            collection_data = self.get_collection_data()
        except Exception:
            collection_data = {"ids": []}

        return {
            "name": self.collection_name,
            "document_count": len(collection_data.get("ids", [])),
            "embedding_dimension": self.embedding_manager.get_embedding_dimension()
        }

    async def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document by its ID."""
        try:
            result = self.get_documents_by_ids(ids=[document_id])
            if result and result.get("ids"):
                doc_index = result["ids"].index(document_id)
                return {
                    "id": result["ids"][doc_index],
                    "text": result["documents"][doc_index],
                    "metadata": result["metadatas"][doc_index],
                }
            return None
        except Exception as e:
            logger.exception("Failed to get document by ID: %s", e)
            return None