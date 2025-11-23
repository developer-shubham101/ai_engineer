# 2–3 lines BEFORE file content (file header)
# app/services/chroma_utils.py
# Utilities that encapsulate ChromaDB client/collection operations
# Use these helpers from other service modules to keep DB logic centralized.

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import chromadb
    from chromadb.config import Settings
except Exception:
    chromadb = None  # will raise at runtime if used without installation

logger = logging.getLogger(__name__)

# Import centralized paths and constants
from app.services.utility import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
)


def ensure_chroma_client(persist_directory: Optional[str] = None, collection_name: Optional[str] = None) -> Tuple[Any, Any]:
    """
    Initialize (or return cached) chroma client and collection.
    Returns (client, collection).
    Tries a couple of common initialization styles to be resilient across chromadb versions.
    """
    global chromadb
    if chromadb is None:
        raise ImportError("chromadb is not installed. Install chromadb to use local RAG.")

    persist_directory = persist_directory or str(DEFAULT_PERSIST_DIR)
    if collection_name is None:
        collection_name = DEFAULT_COLLECTION_NAME

    # Try new API first, fallback to older APIs
    try:
        from chromadb.config import Settings as _Settings
        client = chromadb.Client(_Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
    except Exception:
        try:
            client = chromadb.PersistentClient(path=str(persist_directory))
        except Exception as e:
            logger.exception("Failed to initialize chroma client: %s", e)
            raise

    try:
        collection = client.get_or_create_collection(name=collection_name)
    except Exception:
        # fallback: try without options
        collection = client.get_or_create_collection(name=collection_name)
    return client, collection


def add_documents_to_collection(collection: Any,
                                documents: List[str],
                                metadatas: List[Dict[str, Any]],
                                ids: List[str],
                                embeddings: Optional[List[List[float]]] = None) -> None:
    """
    Add documents to an existing collection. Handles different chroma method signatures.
    """
    logger.warning("Mark:- add_documents_to_collection called where documents=%s, metadatas=%s, ids=%s, embeddings=%s",
                 documents, metadatas, ids, "provided" if embeddings else "not provided")

    try:
        if embeddings is not None:
            collection.add(documents=documents, metadatas=metadatas, ids=ids, embeddings=embeddings)
        else:
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
        logger.info("Added %d documents to collection", len(documents))
        return
    except Exception as e:
        logger.exception("collection.add failed: %s", e)
        # Try fallback variations if any
        try:
            # Some versions accept "documents" + "metadatas" only
            collection.add(documents=documents, metadatas=metadatas, ids=ids)
            return
        except Exception:
            logger.exception("Fallback collection.add also failed")
            raise


def query_collection(collection: Any,
                     query_embeddings: Optional[List[List[float]]] = None,
                     query_texts: Optional[List[str]] = None,
                     n_results: int = 3) -> Dict[str, Any]:
    """
    Query the collection by embeddings or text. Returns the raw result object/dict.
    """
    try:
        if query_embeddings is not None:
            return collection.query(query_embeddings=query_embeddings, n_results=n_results)
        if query_texts is not None:
            return collection.query(query_texts=query_texts, n_results=n_results)
        raise ValueError("Either query_embeddings or query_texts must be provided")
    except Exception as e:
        logger.exception("query_collection failed: %s", e)
        # re-raise so caller can fallback if needed
        raise


def get_collection_data(collection: Any) -> Dict[str, Any]:
    """
    Try to return a dict-like snapshot of the collection (ids/documents/metadatas).
    Behavior depends on chroma client version; this function normalizes common outputs.
    """
    try:
        data = collection.get()
        if isinstance(data, dict):
            return data
        # If it's an object, try to map attributes
        result = {}
        if hasattr(data, "ids"):
            result["ids"] = data.ids
        if hasattr(data, "documents"):
            result["documents"] = data.documents
        if hasattr(data, "metadatas"):
            result["metadatas"] = data.metadatas
        return result
    except Exception as e:
        logger.exception("get_collection_data failed: %s", e)
        raise


def get_documents_by_ids(collection: Any, ids: List[str]) -> Dict[str, Any]:
    """
    Get documents/metadatas for the provided ids (if supported by client).
    """
    try:
        return collection.get(ids=ids)
    except Exception as e:
        logger.exception("get_documents_by_ids failed: %s", e)
        raise


def update_metadatas(collection: Any, ids: List[str], metadata: Dict[str, Any]) -> bool:
    """
    Update metadata for a list of ids. Returns True on success.
    Tries collection.update(ids=..., metadatas=...) then falls back to per-id re-add if necessary.
    """
    try:
        per_id_metas = [metadata.copy() for _ in ids]
        # First try native update
        collection.update(ids=ids, metadatas=per_id_metas)
        logger.info("Updated metadata for %d ids via collection.update()", len(ids))
        return True
    except Exception:
        logger.debug("collection.update() not supported or failed, trying fallback update")
    # Fallback: attempt to retrieve and re-add with updated meta
    try:
        for i, _id in enumerate(ids):
            try:
                got = collection.get(ids=[_id])
                docs = (got.get("documents") or [[]])[0]
                metas = (got.get("metadatas") or [[]])[0]
                if docs:
                    doc_text = docs[0]
                    old_meta = metas[0] if metas else {}
                    new_meta = {**old_meta, **metadata}
                    collection.add(documents=[doc_text], ids=[_id], metadatas=[new_meta])
            except Exception:
                continue
        logger.info("Fallback metadata update attempted for %d ids", len(ids))
        return True
    except Exception:
        logger.exception("Fallback metadata update failed")
        return False


def delete_ids(collection: Any, ids: List[str]) -> None:
    """
    Delete given ids from the collection (if supported).
    """
    try:
        collection.delete(ids=ids)
        logger.info("Deleted %d ids from collection", len(ids))
    except Exception:
        logger.exception("collection.delete(ids=...) failed")
        raise


def delete_collection_by_name(client: Any, collection_name: str) -> None:
    """
    Delete a collection entirely using the client if supported.
    """
    try:
        client.delete_collection(name=collection_name)
        logger.info("Deleted collection %s via client.delete_collection()", collection_name)
    except Exception:
        logger.exception("client.delete_collection failed for %s", collection_name)
        raise


def delete_all_documents(collection: Any, client: Optional[Any] = None, collection_name: Optional[str] = None) -> None:
    """
    Attempt to remove all documents from the provided collection.
    Tries collection.delete(); falls back to listing ids and deleting them, or deleting collection via client.
    """
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
            all_ids = []
            for ids_list in coll_data.get("ids", []):
                all_ids.extend(ids_list)
            if all_ids:
                collection.delete(ids=all_ids)
                logger.info("Cleared %d ids via delete(ids=...)", len(all_ids))
                return
    except Exception:
        logger.debug("Unable to list ids via collection.get()")

    # last resort: delete collection via client
    if client and collection_name:
        try:
            from app.services.utility import DEFAULT_PERSIST_DIR
            client.delete_collection(name=collection_name)
            # recreate
            ensure_chroma_client(persist_directory=str(DEFAULT_PERSIST_DIR), collection_name=collection_name)
            logger.info("Deleted and recreated collection %s", collection_name)
            return
        except Exception:
            logger.exception("Failed to delete and recreate collection %s via client", collection_name)

    logger.warning("Unable to clear collection using available APIs; collection may still contain documents.")
