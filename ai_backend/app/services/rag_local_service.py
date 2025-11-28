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

from app.config import ENABLE_DYNAMIC_MODEL_SELECTION, DEFAULT_MODEL_NAME
from app.services.base_rag_service import BaseRAGService

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

# Version tracking
from app.services import version_tracking

# Import centralized utilities
from app.services.utility import (
    DEFAULT_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    embed_texts,
    chunk_text_basic,
    sanitize_metadata_dict,
    get_data_path, is_collection_empty,
)
from app.utils.doc_parser import parse_file

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


def _generate_document_id(source_name: str) -> str:
    """Generate a stable document ID from source name."""
    import hashlib
    # Use hash of source name for deterministic document_id
    hash_obj = hashlib.md5(source_name.encode())
    return f"doc_{hash_obj.hexdigest()[:16]}"


def _calculate_next_version(document_id: str) -> str:
    """Calculate the next version number for a document."""
    return version_tracking.generate_next_version(document_id)


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


async def add_document_to_rag_local(
    source_name: str,
    text: str,
    chunks: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None,
    version: Optional[str] = None,
    parent_version: Optional[str] = None,
    status: str = "published",
    version_notes: Optional[str] = None,
    created_by: Optional[str] = None
) -> Dict[str, Any]:
    """
    Add a document (or precomputed chunks) to the local chroma collection with versioning support.

    This function takes a document, splits it into chunks (if not already chunked),
    computes embeddings for each chunk, and then adds the chunks, metadatas,
    and embeddings to the ChromaDB collection. It also tracks version history.

    Args:
        source_name: Name of the source document
        text: Document text content
        chunks: Optional pre-computed chunks
        metadata: Optional metadata dictionary
        document_id: Optional document ID (generated if not provided)
        version: Optional version number (auto-calculated if not provided)
        parent_version: Previous version number
        status: Version status (draft, pending_approval, published, archived)
        version_notes: Optional change description
        created_by: User ID who created this version

    Returns:
        Dictionary with:
        - ids: List of chunk IDs added
        - document_id: Document identifier
        - version: Version number
        - chunk_count: Number of chunks
    """

    # Generate document_id if not provided
    if not document_id:
        document_id = _generate_document_id(source_name)
    
    # Calculate version if not provided
    if not version:
        version = _calculate_next_version(document_id)
    
    if not chunks:
        chunks = chunk_text_basic(text)

    if not chunks:
        logger.warning("No chunks produced for document: %s", source_name)
        return {
            "ids": [],
            "document_id": document_id,
            "version": version,
            "chunk_count": 0
        }

    # sanitize metadata and ensure source is present
    base_meta = metadata or {}
    sanitized_base = sanitize_metadata_dict(base_meta)
    sanitized_base["source"] = source_name
    
    # Add version metadata
    from datetime import datetime
    sanitized_base["document_id"] = document_id
    sanitized_base["version"] = version
    sanitized_base["version_created_at"] = datetime.utcnow().isoformat() + "Z"
    sanitized_base["version_created_by"] = created_by
    sanitized_base["parent_version"] = parent_version
    sanitized_base["status"] = status
    sanitized_base["is_latest_version"] = True  # Will be updated if newer version created
    
    logger.debug("Ingest metadata keys (sample): %s", list(sanitized_base.keys())[:8])
    
    # add ingestion timestamp if not present
    if "ingested_at" not in sanitized_base:
        sanitized_base["ingested_at"] = datetime.utcnow().isoformat() + "Z"

    metadatas = [dict(sanitized_base) for _ in chunks]
    ids = _generate_ids(prefix=f"{document_id}_v{version}", n=len(chunks))
    from app.logging_config import log_user_action, log_sensitive_debug
    
    log_user_action(
        logger, "DOCUMENT_INGESTION_START", created_by,
        source_name=source_name, document_id=document_id, version=version,
        chunk_count=len(chunks), status=status, has_parent=bool(parent_version)
    )
    
    log_sensitive_debug(
        logger, "Document ingestion details",
        ids_sample=ids[:3], metadata_keys=list(sanitized_base.keys()),
        chunk_lengths=[len(c) for c in chunks[:3]]
    )

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
        from app.logging_config import log_performance_metric
        
        log_user_action(
            logger, "DOCUMENT_INGESTION_SUCCESS", created_by,
            source_name=source_name, document_id=document_id, version=version,
            chunk_count=len(chunks), collection=DEFAULT_COLLECTION_NAME
        )
    except Exception as e:
        logger.exception("Failed to add documents to Chroma collection: %s", e)
        raise
    
    # Create version record in version tracking database
    try:
        version_tracking.create_version_record(
            document_id=document_id,
            version=version,
            source_name=source_name,
            chunk_ids=ids,
            created_by=created_by,
            parent_version=parent_version,
            status=status,
            version_notes=version_notes,
            metadata=sanitized_base
        )
        logger.info("Created version record: document_id=%s version=%s", document_id, version)
    except Exception as e:
        logger.warning("Failed to create version record (non-fatal): %s", e)

    # Mark previous version as not latest (if this is an update)
    if parent_version:
        try:
            # Update previous version's is_latest_version flag
            prev_version_info = version_tracking.get_version(document_id, parent_version)
            if prev_version_info:
                prev_chunk_ids = prev_version_info["chunk_ids"]
                update_metadatas(collection=collection, ids=prev_chunk_ids, 
                               metadata={"is_latest_version": False})
                logger.info("Marked previous version %s as not latest", parent_version)
        except Exception as e:
            logger.warning("Failed to update previous version metadata (non-fatal): %s", e)

    return {
        "ids": ids,
        "document_id": document_id,
        "version": version,
        "chunk_count": len(ids)
    }


async def update_document_version(
    document_id: str,
    text: str,
    metadata: Optional[Dict[str, Any]] = None,
    version_notes: Optional[str] = None,
    requester_id: Optional[str] = None,
    status: str = "published"
) -> Dict[str, Any]:
    """
    Create a new version of an existing document (non-destructive update).
    
    Args:
        document_id: Document ID to update
        text: New document content
        metadata: Optional metadata updates
        version_notes: Description of changes
        requester_id: User ID making the update
        status: Version status
    
    Returns:
        Dictionary with version info
    """
    # Get latest version to determine parent
    latest = version_tracking.get_latest_version(document_id)
    if not latest:
        raise ValueError(f"Document {document_id} not found")
    
    parent_version = latest["version"]
    source_name = latest["source_name"]
    
    # Calculate next version
    next_version = _calculate_next_version(document_id)
    
    # Create new version
    result = await add_document_to_rag_local(
        source_name=source_name,
        text=text,
        metadata=metadata,
        document_id=document_id,
        version=next_version,
        parent_version=parent_version,
        status=status,
        version_notes=version_notes,
        created_by=requester_id
    )
    
    from app.logging_config import log_user_action
    log_user_action(
        logger, "DOCUMENT_VERSION_UPDATE", requester_id,
        document_id=document_id, old_version=parent_version, new_version=next_version,
        status=status, has_notes=bool(version_notes)
    )
    return result


async def get_document_version(
    document_id: str,
    version: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """
    Retrieve a specific version of a document with its chunks.
    
    Args:
        document_id: Document ID
        version: Version number (if None, get latest)
    
    Returns:
        Dictionary with chunks, metadata, and version info
    """
    # Get version record
    if version:
        version_info = version_tracking.get_version(document_id, version)
    else:
        version_info = version_tracking.get_latest_version(document_id)
    
    if not version_info:
        return None
    
    # Get chunks from ChromaDB
    try:
        from app.services.chroma_utils import get_documents_by_ids
        client, collection = ensure_chroma_client(
            persist_directory=str(DEFAULT_PERSIST_DIR),
            collection_name=DEFAULT_COLLECTION_NAME
        )
        
        chunk_ids = version_info["chunk_ids"]
        result = get_documents_by_ids(collection, chunk_ids)
        
        chunks = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        
        return {
            "document_id": document_id,
            "version": version_info["version"],
            "source_name": version_info["source_name"],
            "chunks": chunks,
            "metadatas": metadatas,
            "created_at": version_info["created_at"],
            "created_by": version_info["created_by"],
            "status": version_info["status"],
            "version_notes": version_info["version_notes"],
            "parent_version": version_info["parent_version"]
        }
    except Exception as e:
        logger.exception("Failed to retrieve document version: %s", e)
        return None


async def compare_document_versions(
    document_id: str,
    version1: str,
    version2: str
) -> Optional[Dict[str, Any]]:
    """
    Compare two versions of a document.
    
    Args:
        document_id: Document ID
        version1: First version number
        version2: Second version number
    
    Returns:
        Dictionary with version data and diff
    """
    # Get both versions
    v1_data = await get_document_version(document_id, version1)
    v2_data = await get_document_version(document_id, version2)
    
    if not v1_data or not v2_data:
        return None
    
    # Combine chunks into full text
    text1 = "\n\n".join(v1_data["chunks"])
    text2 = "\n\n".join(v2_data["chunks"])
    
    # Compute diff
    import difflib
    diff = difflib.unified_diff(
        text1.splitlines(keepends=True),
        text2.splitlines(keepends=True),
        fromfile=f"Version {version1}",
        tofile=f"Version {version2}",
        lineterm=''
    )
    diff_text = ''.join(diff)
    
    # Calculate statistics
    added_lines = diff_text.count('\n+')
    removed_lines = diff_text.count('\n-')
    chunk_diff = len(v2_data["chunks"]) - len(v1_data["chunks"])
    
    return {
        "document_id": document_id,
        "version1": version1,
        "version2": version2,
        "diff": diff_text,
        "summary": {
            "added_lines": added_lines,
            "removed_lines": removed_lines,
            "chunk_difference": chunk_diff,
            "v1_chunks": len(v1_data["chunks"]),
            "v2_chunks": len(v2_data["chunks"])
        },
        "version1_info": {
            "created_at": v1_data["created_at"],
            "created_by": v1_data["created_by"],
            "notes": v1_data["version_notes"]
        },
        "version2_info": {
            "created_at": v2_data["created_at"],
            "created_by": v2_data["created_by"],
            "notes": v2_data["version_notes"]
        }
    }


async def list_documents(
    department: Optional[str] = None,
    status: Optional[str] = None,
    latest_only: bool = True
) -> List[Dict[str, Any]]:
    """
    List all documents with optional filtering.
    
    Args:
        department: Filter by department
        status: Filter by status
        latest_only: If True, only return latest versions
    
    Returns:
        List of document summaries
    """
    # Get all documents from version tracking
    if status:
        documents = version_tracking.get_documents_by_status(status)
    else:
        documents = version_tracking.list_all_documents(latest_only=latest_only)
    
    # Filter by department if specified
    if department:
        filtered = []
        for doc in documents:
            if doc.get("metadata", {}).get("department") == department:
                filtered.append(doc)
        documents = filtered
    
    return documents


async def archive_document_version(
    document_id: str,
    version: str
) -> bool:
    """
    Archive (soft-delete) a specific version of a document.
    
    Args:
        document_id: Document ID
        version: Version number
    
    Returns:
        True if successful
    """
    try:
        # Update version tracking status
        success = version_tracking.update_version_status(document_id, version, "archived")
        
        if not success:
            return False
        
        # Update ChromaDB metadata
        version_info = version_tracking.get_version(document_id, version)
        if not version_info:
            return False
        
        chunk_ids = version_info["chunk_ids"]
        client, collection = ensure_chroma_client(
            persist_directory=str(DEFAULT_PERSIST_DIR),
            collection_name=DEFAULT_COLLECTION_NAME
        )
        
        update_metadatas(collection=collection, ids=chunk_ids, 
                        metadata={"status": "archived", "is_latest_version": False})
        
        from app.logging_config import log_user_action
        log_user_action(
            logger, "DOCUMENT_VERSION_ARCHIVED", "system",
            document_id=document_id, version=version
        )
        return True
    except Exception as e:
        logger.exception("Failed to archive document version: %s", e)
        return False


class LocalRAGService(BaseRAGService):
    """
    Local RAG service implementation using local LLM models.
    Inherits common functionality from BaseRAGService.
    """
    
    async def generate_response(
        self,
        query_text: str,
        context_text: str,
        final_prefix: str,
        use_llm: bool,
        max_tokens: int,
        session_id: Optional[str]
    ) -> Optional[str]:
        """
        Generate a response using local LLM.
        """
        if not use_llm:
            return None

        if ENABLE_DYNAMIC_MODEL_SELECTION:
            task = "reason"
            model_key = choose_model_for_task(task)
            from app.logging_config import log_user_action
            log_user_action(
                logger, "MODEL_SELECTION_DYNAMIC", "system",
                chosen_model=model_key, task=task, selection_mode="dynamic"
            )
        else:
            model_key = DEFAULT_MODEL_NAME
            from app.logging_config import log_user_action
            log_user_action(
                logger, "MODEL_SELECTION_DEFAULT", "system",
                default_model=DEFAULT_MODEL_NAME, selection_mode="static"
            )

        try:
            llm_instance = get_llm_instance(model_key)
        except Exception as e:
            logger.exception("Failed to load LLM instance: %s", e)
            raise

        # Calculate available tokens for prompt (reserve tokens for generation)
        max_prompt_tokens = 2048 - max_tokens - 50  # 50 token safety margin
        
        prompt = build_prompt_with_selected_chunks(
            prefix=final_prefix,
            context_text=context_text,
            question=query_text,
            max_total_tokens=max_prompt_tokens,
            context_priority=0.65  # Allocate 65% to context, 35% to system/question
        )

        try:
            # Calculate prompt metrics
            prompt_tokens = estimate_tokens_from_text(prompt)
            context_tokens = estimate_tokens_from_text(context_text or "")
            prefix_tokens = estimate_tokens_from_text(final_prefix)
            query_tokens = estimate_tokens_from_text(query_text)
            
            from app.logging_config import log_llm_interaction, log_performance_metric
            import time
            llm_start_time = time.time()
            
            log_llm_interaction(
                logger, "LOCAL_MISTRAL", prompt_tokens, 0,  # response tokens unknown yet
                model_key=model_key, prompt_len=len(prompt), max_tokens=max_tokens,
                session_id=session_id or "none"
            )
            
            logger.info("LOCAL_PROMPT_METRICS: prefix_tokens=%d context_tokens=%d query_tokens=%d total_tokens=%d",
                       prefix_tokens, context_tokens, query_tokens, prompt_tokens)
            
            from app.logging_config import log_sensitive_debug
            log_sensitive_debug(
                logger, "Local LLM prompt components",
                final_prefix=final_prefix, context_text=context_text or "[NO_CONTEXT]",
                query_text=query_text, model_key=model_key
            )
            
            log_sensitive_debug(
                logger, "Local LLM full prompt",
                full_prompt=prompt, prompt_len=len(prompt), prompt_tokens=prompt_tokens
            )
            
            # Check if prompt might exceed context window
            estimated_total = prompt_tokens + max_tokens
            context_window_limit = 2048  # Common context window size
            if estimated_total > context_window_limit:
                from app.logging_config import log_security_event
                log_security_event(
                    logger, "POTENTIAL_CONTEXT_OVERFLOW", "system",
                    estimated_total=estimated_total, prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens, context_limit=context_window_limit,
                    model_key=model_key, session_id=session_id
                )
            
            answer = await _call_llm_with_retry(
                llm_instance,
                prompt,
                max_tokens=max_tokens,
                temperature=0.0
            )
            
            response_len = len(answer or "")
            response_tokens = estimate_tokens_from_text(answer or "")
            
            llm_duration = (time.time() - llm_start_time) * 1000
            
            log_llm_interaction(
                logger, "LOCAL_MISTRAL", prompt_tokens, response_tokens,
                model_key=model_key, response_len=response_len, 
                duration_ms=llm_duration, session_id=session_id or "none"
            )
            
            log_performance_metric(
                logger, "LOCAL_LLM_GENERATION", llm_duration,
                model_key=model_key, prompt_tokens=prompt_tokens, 
                response_tokens=response_tokens, session_id=session_id
            )
            
            log_sensitive_debug(
                logger, "Local LLM response",
                response_text=answer or "", response_len=response_len,
                response_tokens=response_tokens
            )
            
            # Log efficiency metrics
            efficiency_ratio = response_tokens / max(prompt_tokens, 1)
            tokens_per_second = response_tokens / max(llm_duration / 1000, 0.001)
            
            logger.info("LOCAL_EFFICIENCY_METRICS: input_tokens=%d output_tokens=%d efficiency_ratio=%.2f tokens_per_sec=%.1f",
                       prompt_tokens, response_tokens, efficiency_ratio, tokens_per_second)
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            raise

        return answer


# Create global instance
_local_rag_service = LocalRAGService()


async def query_local_rag(
        query_text: str,
        n_results: int = 3,
        requester: Optional[Dict[str, str]] = None,
        llm_prompt_prefix: Optional[str] = None,
        use_llm: bool = True,
        max_tokens: int = 256,
        session_id: Optional[str] = None,
        model_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query the local RAG service using the base RAG functionality.
    """
    # Store model_key in service for generate_response method
    _local_rag_service._model_key = model_key
    return await _local_rag_service.query_rag(
        query_text=query_text,
        n_results=n_results,
        requester=requester,
        llm_prompt_prefix=llm_prompt_prefix,
        use_llm=use_llm,
        max_tokens=max_tokens,
        session_id=session_id
    )


async def seed_from_file(file_path: Optional[str] = None, source_name: Optional[str] = None,
                         force_reseed: bool = False) -> List[str]:
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
    default_path = get_data_path("company")
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
        SHOW_DATA = True # Just for debugging purpose
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

    #If path is a directory, check if it contains version subdirectories
    if path.is_dir():
        # NOTE: If force_reseed is True, we are re-adding chunks which may result in duplicates
        # unless IDs are managed carefully. For a learning environment, this is often acceptable
        # for a "refresh" or requires a full collection clear, which is a separate endpoint.
        # We will log a warning if reseed is forced.
        if force_reseed:
            logger.warning("Force re-seeding entire directory: %s. This may create duplicate chunks.", path)

        logger.info("Seeding directory: %s", path)
        
        # Check if this directory contains version subdirectories (v1, v2, v3, etc.)
        # Filter for directories starting with 'v' followed by a number
        version_dirs = []
        for d in path.iterdir():
            if d.is_dir() and d.name.lower().startswith('v'):
                try:
                    # Verify the rest is a number (e.g., "1", "1.0", "2.5")
                    float(d.name[1:])
                    version_dirs.append(d)
                except ValueError:
                    continue
        
        if version_dirs:
            # Sort version directories numerically (v2 < v10)
            def get_version_float(d):
                try:
                    return float(d.name[1:])
                except ValueError:
                    return 0.0
            
            sorted_version_dirs = sorted(version_dirs, key=get_version_float)
            logger.info("Found %d version directories in %s. Processing order: %s", 
                       len(sorted_version_dirs), path, [d.name for d in sorted_version_dirs])
            
            category = path.name  # e.g., "company", "mission", etc.
            
            # Track the last seen version for each document to correctly link parents
            # Map: document_base_name -> version_string
            latest_versions_map = {}
            
            for version_dir in sorted_version_dirs:
                version_str = version_dir.name[1:]  # Remove 'v' prefix
                # Normalize version to semantic format (e.g., "1" -> "1.0")
                if '.' not in version_str:
                    version_str = f"{version_str}.0"
                
                logger.info("Processing version directory: %s (version %s)", version_dir.name, version_str)
                
                for file_path in sorted(version_dir.iterdir()):
                    # Skip .meta.json files (they're companions, not documents)
                    if file_path.suffix == '.json' and file_path.stem.endswith('.meta'):
                        continue
                        
                    if file_path.is_file():
                        try:
                            # Use doc_parser to read and parse file
                            text = parse_file(str(file_path))
                            
                            # Generate document_id based on category + filename (same across versions)
                            doc_base_name = file_path.stem  # filename without extension
                            document_id = _generate_document_id(f"{category}/{doc_base_name}")
                            
                            # Source name for display
                            src_name = f"{category}/{version_dir.name}/{file_path.name}"
                            
                            # Load custom metadata from companion .meta.json file
                            meta_file = file_path.with_suffix('.meta.json')
                            custom_meta = {}
                            if meta_file.exists():
                                try:
                                    import json
                                    custom_meta = json.loads(meta_file.read_text(encoding='utf-8'))
                                    logger.info("Loaded metadata from %s", meta_file.name)
                                except Exception as e:
                                    logger.warning("Failed to load metadata from %s: %s", meta_file.name, e)
                            
                            # Merge metadata: custom metadata takes precedence
                            metadata = {
                                "seeded": True,
                                "category": category,
                                **custom_meta  # Merge custom metadata
                            }
                            
                            # Determine parent version dynamically
                            # If we've seen this doc before, that's the parent. 
                            # If not, and this isn't v1.0, parent is None (it's a new doc introduced in a later version)
                            parent_version = latest_versions_map.get(doc_base_name)
                            
                            result = await add_document_to_rag_local(
                                source_name=src_name,
                                text=text,
                                chunks=None,
                                metadata=metadata,  # Use merged metadata
                                document_id=document_id,
                                version=version_str,
                                parent_version=parent_version,
                                status="published",
                                created_by="system_seed"
                            )
                            
                            if result and result.get("ids"):
                                added_ids.extend(result["ids"])
                                from app.logging_config import log_user_action
                            log_user_action(
                                logger, "SEED_FILE_PROCESSED", "system_seed",
                                filename=file_path.name, version=version_str, 
                                chunk_count=result["chunk_count"], document_id=document_id,
                                parent_version=parent_version, category=category
                            )
                                
                            # Update the map so the next version knows this is the parent
                            latest_versions_map[doc_base_name] = version_str
                            
                        except Exception as e:
                            logger.exception("Failed to seed file %s: %s", file_path, e)
                            continue
        else:
            # Backward compatibility: process files directly in directory (old behavior)
            logger.info("No version directories found, processing files directly")
            for child in sorted(path.iterdir()):
                if child.is_file():
                    try:
                        # Use doc_parser to read and parse file
                        text = parse_file(str(child))
                        # Use relative path + name for source_name for better uniqueness
                        src_name = str(child.relative_to(path.parent))
                        result = await add_document_to_rag_local(source_name=src_name, text=text, chunks=None,
                                                               metadata={"seeded": True})
                        if result and result.get("ids"):
                            added_ids.extend(result["ids"])
                            from app.logging_config import log_user_action
                            log_user_action(
                                logger, "SEED_FILE_LEGACY", "system_seed",
                                filename=child.name, chunk_count=result["chunk_count"],
                                version=result["version"]
                            )
                    except Exception as e:
                        logger.exception("Failed to seed file %s: %s", child, e)
                        continue
        return added_ids

    # Otherwise, it's a single file; ingest it. (Old behavior, primarily for backward compatibility)
    try:
        text = parse_file(str(path))
    except Exception as e:
        logger.exception("Failed to read seed file %s: %s", path, e)
        return []

    name = source_name or path.name
    try:
        result = await add_document_to_rag_local(source_name=name, text=text, chunks=None, metadata={"seeded": True})
        if result and result.get("ids"):
            added_ids.extend(result["ids"])
            from app.logging_config import log_user_action
            log_user_action(
                logger, "SEED_SINGLE_FILE", "system_seed",
                filename=path.name, chunk_count=result["chunk_count"],
                version=result["version"]
            )
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
