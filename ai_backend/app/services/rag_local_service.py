# app/services/rag_local_service.py
"""
This module provides a local Retrieval-Augmented Generation (RAG) service.

It handles:
- Initialization of local RAG resources (ChromaDB, embedding models, LLMs).
- Adding documents to the local RAG knowledge base.
- Querying the local RAG with RBAC filtering and tone-aware guidance.
- Seeding the knowledge base from default files/directories.
- Helper functions for model routing, token budgeting, and prompt construction.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

# new import to fetch recent messages (tone is stored there by support_chat)
from app.services.support_chat import fetch_recent_messages

# Embed/LLM imports (optional at runtime)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from langchain.llms import LlamaCpp
except Exception:
    LlamaCpp = None

# Chroma utils (centralized DB helpers)
from app.services.chroma_utils import (
    ensure_chroma_client,
    add_documents_to_collection,
    query_collection,
    get_collection_data,
    update_metadatas,
    delete_all_documents,
)

# Import centralized utilities
from app.config import (
    ENABLE_DYNAMIC_MODEL_SELECTION,
    DEFAULT_MODEL_NAME,
    DEFAULT_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
)
from app.services.utility import (
    embed_texts,
    chunk_text_basic,
    sanitize_metadata_dict,
    get_data_path, is_collection_empty,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

from app.services.model_manager import choose_model_for_task, get_llm_instance
from app.services.prompt_builder import (
    estimate_tokens_from_text,
    build_prompt_with_selected_chunks,
    _call_llm_with_retry,
    build_tone_guidance,
)

# Internal global handles - model cache
_llm_instances = {}  # Dict[str, Any] - cache for different model keys


# ---------- Utilities ----------

def _generate_ids(prefix: str, n: int) -> List[str]:
    """Generate a list of unique IDs with a given prefix."""
    return [f"{prefix}_{uuid.uuid4().hex}" for _ in range(n)]


# ---------- Public API ----------

def initialize_local_rag(embedding_model_instance: Optional[Any] = None,
                         llm_instance: Optional[Any] = None,
                         persist_directory: Optional[str] = None,
                         collection_name: Optional[str] = None) -> None:
    """
    Initialize resources for the local RAG service.

    This function ensures that the ChromaDB client and collection are ready,
    and it can optionally accept pre-initialized embedding and LLM instances.
    LLM instances are now managed via get_llm_instance() with model routing.

    Args:
        embedding_model_instance: An optional pre-initialized embedding model instance.
        llm_instance: An optional pre-initialized LLM instance.
        persist_directory: The directory to persist ChromaDB data.
        collection_name: The name of the ChromaDB collection to use.
    """
    global _llm_instances

    if embedding_model_instance is not None:
        # Note: The shared embedding model instance is managed in utility.py
        # If a custom instance is provided, it would need to be set there
        logger.warning("Custom embedding_model_instance provided but shared instance is used from utility.py")

    if llm_instance is not None:
        # Store as default "mistral" model (backward compatibility)
        _llm_instances["mistral"] = llm_instance
        logger.info("Using provided LLM instance (stored as 'mistral' key)")
    else:
        logger.info("No LLM instance provided; local LLM will be lazy-loaded on demand via model router")

    # Ensure Chroma client & collection exist
    ensure_chroma_client(persist_directory=str(persist_directory or DEFAULT_PERSIST_DIR),
                         collection_name=collection_name or DEFAULT_COLLECTION_NAME)
    logger.info("Local RAG initialization completed (collection: %s)", collection_name or DEFAULT_COLLECTION_NAME)


async def add_document_to_rag_local(source_name: str,
                                    text: str,
                                    chunks: Optional[List[str]] = None,
                                    metadata: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Add a document (or precomputed chunks) to the local chroma collection.

    This function takes a document, splits it into chunks (if not already chunked),
    computes embeddings for each chunk, and then adds the chunks, metadatas,
    and embeddings to the ChromaDB collection.

    Returns the list of ids added.

    - Splits text into chunks if chunks not provided.
    - Computes embeddings locally for each chunk.
    - Adds documents, metadatas, ids, and embeddings to Chroma via chroma_utils.
    """

    if not chunks:
        chunks = chunk_text_basic(text)

    if not chunks:
        logger.warning("No chunks produced for document: %s", source_name)
        return []

    # sanitize metadata and ensure source is present
    base_meta = metadata or {}
    sanitized_base = sanitize_metadata_dict(base_meta)
    sanitized_base["source"] = source_name
    logger.debug("Ingest metadata keys (sample): %s", list(sanitized_base.keys())[:8])
    # add ingestion timestamp if not present
    if "ingested_at" not in sanitized_base:
        from datetime import datetime
        sanitized_base["ingested_at"] = datetime.utcnow().isoformat() + "Z"

    metadatas = [dict(sanitized_base) for _ in chunks]
    ids = _generate_ids(prefix=source_name, n=len(chunks))
    logger.info("Preparing to add document to RAG: source=%s chunks=%d ids_sample=%s", source_name, len(chunks),
                ids[:3])

    # compute embeddings locally
    try:
        embeddings = await embed_texts(chunks)
    except Exception as e:
        logger.exception("Failed to compute embeddings locally: %s", e)
        raise

    # Add to chroma via helper
    try:
        client, collection = ensure_chroma_client(persist_directory=str(DEFAULT_PERSIST_DIR),
                                                  collection_name=DEFAULT_COLLECTION_NAME)
        add_documents_to_collection(collection=collection, documents=chunks, metadatas=metadatas, ids=ids,
                                    embeddings=embeddings)
        logger.info("Added %d chunks for source %s to collection %s", len(chunks), source_name, DEFAULT_COLLECTION_NAME)
    except Exception as e:
        logger.exception("Failed to add documents to Chroma collection: %s", e)
        raise

    return ids


async def query_local_rag(
        query_text: str,
        n_results: int = 3,
        requester: Optional[Dict[str, str]] = None,
        llm_prompt_prefix: Optional[str] = None,
        use_llm: bool = True,
        max_tokens: int = 256,
        session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query the local RAG service.

    This is the main function for querying the RAG. It performs the following steps:
    1. Computes an embedding for the user's query.
    2. Retrieves the top-k most relevant documents from ChromaDB.
    3. Applies Role-Based Access Control (RBAC) to filter the retrieved documents.
    4. Injects tone-aware guidance into the LLM prompt prefix.
    5. If `use_llm` is True, it calls the LLM with the constructed prompt to generate an answer.
    6. Returns a dictionary containing the answer, retrieved documents, and other metadata.

    Args:
        query_text: The user's query.
        n_results: The number of documents to retrieve.
        requester: A dictionary containing information about the user making the request (for RBAC).
        llm_prompt_prefix: A prefix to add to the LLM prompt.
        use_llm: Whether to use the LLM to generate an answer.
        max_tokens: The maximum number of tokens for the LLM to generate.
        session_id: The ID of the user's session (for fetching conversation history and tone).

    Returns:
        A dictionary containing the RAG output.
    """
    # Ensure Chroma client
    client, collection = ensure_chroma_client(
        persist_directory=str(DEFAULT_PERSIST_DIR),
        collection_name=DEFAULT_COLLECTION_NAME
    )

    logger.debug(
        "query_local_rag called: query_text_len=%d n_results=%d use_llm=%s max_tokens=%d session_id=%s requester=%s",
        len(query_text or ""), n_results, use_llm, max_tokens, session_id, (requester or {}).get("user_id"))

    if not query_text:
        raise ValueError("query_text must be provided")

    # -----------------------------
    # 1. Get embedding for query
    # -----------------------------
    try:
        q_emb = (await embed_texts([query_text]))[0]
        logger.debug("Computed query embedding.")
    except Exception as e:
        logger.exception("Failed to embed query: %s", e)
        raise

    # -----------------------------
    # 2. Retrieve from Chroma
    # -----------------------------
    try:
        result = query_collection(collection=collection, query_embeddings=[q_emb], n_results=n_results)
    except Exception:
        # fallback to text search
        result = query_collection(collection=collection, query_texts=[query_text], n_results=n_results)

    # Normalize shapes
    if isinstance(result, dict):
        raw_docs = (result.get("documents") or [[]])[0]
        raw_metadatas = (result.get("metadatas") or [[]])[0]
        raw_ids = (result.get("ids") or [[]])[0]
        raw_distances = (result.get("distances") or [[]])[0]
    else:
        try:
            raw_docs = result.documents[0]
            raw_metadatas = result.metadatas[0]
            raw_ids = result.ids[0]
            raw_distances = result.distances[0] if hasattr(result, "distances") else []
        except Exception as e:
            logger.exception("Unexpected Chroma format: %s", e)
            raw_docs, raw_metadatas, raw_ids, raw_distances = [], [], [], []

    try:
        logger.debug("Raw retrieval counts: docs=%d metadatas=%d ids=%d distances=%d",
                     len(raw_docs), len(raw_metadatas), len(raw_ids), len(raw_distances))
    except Exception:
        logger.debug("Raw retrieval: unable to compute counts (unexpected shape)")

    # ------------------------------------------
    # 3. RBAC filtering (visible vs filtered)
    # ------------------------------------------
    def _allowed_by_metadata(meta: Optional[Dict[str, Any]], requester: Optional[Dict[str, str]]) -> bool:
        """Check if a document is accessible based on its metadata and the requester's role/department."""
        sens = meta.get("sensitivity", "public_internal") if meta else "public_internal"

        # personal
        if sens == "personal":
            owner = meta.get("owner_id")
            if requester and owner == requester.get("user_id"):
                return True
            return requester and requester.get("role") in ("HR", "Legal", "Executive")

        # highly_confidential
        if sens == "highly_confidential":
            return requester and requester.get("role") in ("Legal", "Executive")

        # role_confidential
        if sens == "role_confidential":
            allowed_roles = meta.get("allowed_roles") or []
            if requester and requester.get("role") in allowed_roles:
                return True
            return requester and requester.get("role") in ("HR", "Legal", "Executive")

        # department_confidential
        if sens == "department_confidential":
            if requester and requester.get("department") == meta.get("department"):
                return True
            return requester and requester.get("role") in ("HR", "Legal", "Executive")

        # public_internal
        return True

    visible_docs, visible_metas, visible_ids, visible_distances = [], [], [], []
    public_summaries, filtered_details = [], []
    filtered_out_count = 0

    for doc, meta, id_, dist in zip(raw_docs, raw_metadatas, raw_ids, raw_distances):
        try:
            if _allowed_by_metadata(meta, requester):
                visible_docs.append(doc)
                visible_metas.append(meta)
                visible_ids.append(id_)
                visible_distances.append(dist)
            else:
                filtered_out_count += 1
                ps = meta.get("public_summary") if isinstance(meta, dict) else None
                if ps:
                    public_summaries.append(ps)
                filtered_details.append({
                    "id": id_,
                    "sensitivity": meta.get("sensitivity"),
                    "department": meta.get("department"),
                    "source": meta.get("source"),
                })
        except Exception as e:
            logger.exception("Metadata filtering error: %s", e)

    # ------------------------------------------
    # 4. Build Context
    # ------------------------------------------
    context_text = "\n\n---\n\n".join(visible_docs or [])

    logger.info("Post-filtering: visible_docs=%d filtered_out=%d", len(visible_docs), filtered_out_count)

    out: Dict[str, Any] = {
        "documents": visible_docs,
        "metadatas": visible_metas,
        "ids": visible_ids,
        "distances": visible_distances,
        "raw_documents": raw_docs,
        "raw_metadatas": raw_metadatas,
        "raw_ids": raw_ids,
        "raw_distances": raw_distances,
        "context": context_text,
        "filtered_out_count": filtered_out_count,
        "public_summaries": public_summaries,
        "filtered_details": filtered_details,
    }

    # ------------------------------------------
    # 5. Tone-Based Prefix Injection
    # ------------------------------------------
    last_user_tone = None
    if session_id:
        try:
            history = fetch_recent_messages(session_id, limit=10)
            for m in reversed(history):
                if m.get("speaker") == "user" and m.get("tone"):
                    last_user_tone = m["tone"]
                    break
        except Exception as e:
            logger.warning("Tone fetch failed: %s", e)

    logger.debug("Last user tone detected: %s", last_user_tone)

    tone_note = build_tone_guidance(last_user_tone)

    # Build LLM prefix
    system_prefix = llm_prompt_prefix or (
        "You are a helpful assistant. Use the provided context to answer the question. "
        "If the answer is not present in the context, say you don't know."
    )

    final_prefix = (
        f"Conversation Tone Guidance:\n{tone_note}\n\n"
        f"{system_prefix}"
    )

    # Log approx sizes for debugging
    try:
        approx_prefix_tokens = estimate_tokens_from_text(final_prefix)
        approx_context_tokens = estimate_tokens_from_text(context_text)
        logger.debug("Prompt sizes: prefix_chars=%d context_chars=%d est_prefix_tokens=%d est_context_tokens=%d",
                     len(final_prefix), len(context_text), approx_prefix_tokens, approx_context_tokens)
    except Exception:
        logger.debug("Failed to estimate prompt sizes")

    # ------------------------------------------
    # 6. LLM Call with Model Routing
    # ------------------------------------------
    if use_llm:
        # Use dynamic model selection only if enabled, otherwise use default model
        if ENABLE_DYNAMIC_MODEL_SELECTION:
            # Choose model based on task type (default to "reason" for RAG)
            task = "reason"  # Could be made configurable via parameter
            model_key = choose_model_for_task(task)
            logger.info("Model chosen=%s for task=%s (dynamic selection enabled)", model_key, task)
        else:
            # Use default model (mistral-7b-instruct-v0.2.Q3_K_M)
            model_key = "default"
            logger.info("Using default model: %s", DEFAULT_MODEL_NAME)

        try:
            llm_instance = get_llm_instance(model_key)
        except Exception as e:
            logger.exception("Failed to load LLM instance: %s", e)
            raise

        prompt = build_prompt_with_selected_chunks(final_prefix, context_text, query_text)

        try:
            answer = await _call_llm_with_retry(
                llm_instance,
                prompt,
                max_tokens=max_tokens,
                temperature=0.0
            )
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            raise

        try:
            answer_len = len(str(answer)) if answer is not None else 0
        except Exception:
            answer_len = 0
        logger.info("LLM returned answer (approx length=%d) for query session=%s", answer_len, session_id)

        out["answer"] = answer

    return out


async def seed_from_file(file_path: Optional[str] = None, source_name: Optional[str] = None, force_reseed: bool = False) -> \
List[str]:
    """
    Read the given file or directory and index it.

    This function seeds the ChromaDB collection with documents from a file or directory.
    It has a "load once" logic to avoid re-seeding on every application startup.

    Behavior:
    - If file_path is None: attempts to seed from default project data/companyData directory.
    - If collection is NOT empty AND force_reseed is False, it skips seeding.
    - If file_path is a file: read & ingest that single file.
    - If file_path is a directory: iterate non-recursively through files in the directory
      and ingest each file found (skip directories). Returns a flat list of all chunk ids added.

    Returns list of ids added (may be empty).
    """

    # NEW DEFAULT PATH: data/companyData
    default_path = get_data_path("companyData")
    path = Path(file_path) if file_path else default_path
    logger.info("looking for path for data %s", path)
    if not path.exists():
        logger.warning("Seed path not found at %s", path)
        return []

    # Check collection size for "load once" logic on startup
    client, collection = ensure_chroma_client(persist_directory=str(DEFAULT_PERSIST_DIR),
                                              collection_name=DEFAULT_COLLECTION_NAME)
    try:
        data = get_collection_data(collection)
        SHOW_DATA = True
        if SHOW_DATA:
            from rich import print as rprint
            rprint(data.get("ids"))
        has_data = not is_collection_empty(data)
    except Exception as e:
        logger.warning("Could not check collection size: %s. Assuming zero.", e)
        has_data = False
        data = {"ids": []}

    logger.info("has_data %s", has_data)
    if has_data and not force_reseed:
        logger.info(
            "Collection already contains %d documents. Skipping seed on startup (use /seed?reseed=true to force).",
            len(data.get("ids")))
        return []

    added_ids: List[str] = []

    # If path is a directory, iterate files (non-recursive) and ingest each
    if path.is_dir():
        # NOTE: If force_reseed is True, we are re-adding chunks which may result in duplicates
        # unless IDs are managed carefully. For a learning environment, this is often acceptable
        # for a "refresh" or requires a full collection clear, which is a separate endpoint.
        # We will log a warning if reseed is forced.
        if force_reseed:
            logger.warning("Force re-seeding entire directory: %s. This may create duplicate chunks.", path)

        logger.info("Seeding directory: %s", path)
        for child in sorted(path.iterdir()):
            if child.is_file():
                try:
                    text = child.read_text(encoding="utf-8")
                    # Use relative path + name for source_name for better uniqueness
                    src_name = str(child.relative_to(path.parent))
                    ids = await add_document_to_rag_local(source_name=src_name, text=text, chunks=None,
                                                    metadata={"seeded": True})
                    if ids:
                        added_ids.extend(ids)
                        logger.info("Seeded file %s -> %d chunks", child.name, len(ids))
                except Exception as e:
                    logger.exception("Failed to seed file %s: %s", child, e)
                    continue
        return added_ids

    # Otherwise, it's a single file; ingest it. (Old behavior, primarily for backward compatibility)
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.exception("Failed to read seed file %s: %s", path, e)
        return []

    name = source_name or path.name
    try:
        ids = await add_document_to_rag_local(source_name=name, text=text, chunks=None, metadata={"seeded": True})
        if ids:
            added_ids.extend(ids)
            logger.info("Seeded file %s -> %d chunks", path.name, len(ids))
    except Exception as e:
        logger.exception("Failed to seed file %s: %s", path, e)

    return added_ids


def update_metadata(ids: List[str], metadata: Dict[str, Any]) -> bool:
    """
    Wrapper that updates metadata for existing ids using chroma_utils.update_metadatas.
    """
    client, collection = ensure_chroma_client(persist_directory=str(DEFAULT_PERSIST_DIR),
                                              collection_name=DEFAULT_COLLECTION_NAME)
    sanitized = sanitize_metadata_dict(metadata)
    return update_metadatas(collection=collection, ids=ids, metadata=sanitized)


def clear_collection() -> None:
    """
    Delete all documents from the collection. Use with caution.
    """
    client, collection = ensure_chroma_client(persist_directory=str(DEFAULT_PERSIST_DIR),
                                              collection_name=DEFAULT_COLLECTION_NAME)
    try:
        delete_all_documents(collection=collection, client=client, collection_name=DEFAULT_COLLECTION_NAME)
    except Exception as e:
        logger.exception("Error clearing collection: %s", e)
        raise
