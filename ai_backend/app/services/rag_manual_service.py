from __future__ import annotations
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import centralized utilities
from app.services.utility import (
    BASE_DIR,
    DEFAULT_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    get_embedding_model_instance,
    embed_texts,
    chunk_text_basic,
    sanitize_metadata_dict,
    MODELS_DIR,
)

# Optional local LLM support
try:
    from langchain.llms import LlamaCpp
except Exception:
    LlamaCpp = None

# Centralized chroma helpers
from app.services.chroma_utils import (
    ensure_chroma_client,
    add_documents_to_collection,
    query_collection,
    get_collection_data,
    get_documents_by_ids,
    update_metadatas,
    delete_all_documents,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


_llm_instance = None


# -------------------
# Utilities (now imported from utility.py)
# -------------------
# All utility functions are now imported from utility.py


# -------------------
# Public/simple API
# -------------------
def add_document_manual(source_name: str, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Simpler wrapper to add a document to Chroma:
      - chunks text
      - computes embeddings locally
      - writes chunks + metadatas + ids to Chroma via chroma_utils
    Returns list of chunk ids created.
    """
    if not text:
        logger.warning("add_document_manual called with empty text for %s", source_name)
        return []

    chunks = chunk_text_basic(text)
    if not chunks:
        logger.warning("No chunks produced for %s", source_name)
        return []

    base_meta = metadata or {}
    sanitized_base = sanitize_metadata_dict(base_meta)
    sanitized_base["source"] = source_name
    if "ingested_at" not in sanitized_base:
        from datetime import datetime
        sanitized_base["ingested_at"] = datetime.utcnow().isoformat() + "Z"

    metadatas = [dict(sanitized_base) for _ in chunks]
    ids = [f"{source_name}_{uuid.uuid4().hex}" for _ in chunks]

    try:
        embeddings = embed_texts(chunks)
    except Exception as e:
        logger.exception("Failed to embed chunks for %s: %s", source_name, e)
        raise

    client, collection = ensure_chroma_client(persist_directory=str(DEFAULT_PERSIST_DIR), collection_name=DEFAULT_COLLECTION_NAME)
    add_documents_to_collection(collection=collection, documents=chunks, metadatas=metadatas, ids=ids, embeddings=embeddings)
    logger.info("Manually added %d chunks for %s", len(ids), source_name)
    return ids


def query_manual_rag(query_text: str,
                     n_results: int = 3,
                     requester: Optional[Dict[str, str]] = None,
                     use_llm: bool = False,
                     max_tokens: int = 256) -> Dict[str, Any]:
    """
    Simple manual query function:
      - retrieves top-k docs from Chroma (raw)
      - applies the same simple RBAC filtering used elsewhere (keeps API consistent)
      - returns visible docs + public_summaries + filtered count etc.
    """
    client, collection = ensure_chroma_client(persist_directory=str(DEFAULT_PERSIST_DIR), collection_name=DEFAULT_COLLECTION_NAME)

    if not query_text:
        raise ValueError("query_text must be provided")

    # embed
    q_emb = embed_texts([query_text])[0]

    # query via helper (prefers embeddings)
    try:
        result = query_collection(collection=collection, query_embeddings=[q_emb], n_results=n_results)
    except Exception:
        result = query_collection(collection=collection, query_texts=[query_text], n_results=n_results)

    # normalize
    if isinstance(result, dict):
        raw_docs = (result.get("documents") or [[]])[0]
        raw_metas = (result.get("metadatas") or [[]])[0]
        raw_ids = (result.get("ids") or [[]])[0]
        raw_dists = (result.get("distances") or [[]])[0]
    else:
        try:
            raw_docs = result.documents[0]
            raw_metas = result.metadatas[0]
            raw_ids = result.ids[0]
            raw_dists = result.distances[0] if hasattr(result, "distances") else []
        except Exception as e:
            logger.exception("Unexpected query result format: %s", e)
            raw_docs, raw_metas, raw_ids, raw_dists = [], [], [], []

    # RBAC helper (same logic as main service)
    def _allowed(meta: Optional[Dict[str, Any]], requester: Optional[Dict[str, str]]) -> bool:
        if not meta:
            return requester is not None
        sens = meta.get("sensitivity", "public_internal")
        if sens == "personal":
            owner = meta.get("owner_id")
            if owner == (requester.get("user_id") if requester else None):
                return True
            return requester and requester.get("role") in ("HR", "Legal", "Executive")
        if sens == "highly_confidential":
            return requester and requester.get("role") in ("Legal", "Executive")
        if sens == "role_confidential":
            allowed = meta.get("allowed_roles") or []
            if requester and requester.get("role") in allowed:
                return True
            return requester and requester.get("role") in ("HR", "Legal", "Executive")
        if sens == "department_confidential":
            if requester and requester.get("department") == meta.get("department"):
                return True
            return requester and requester.get("role") in ("HR", "Legal", "Executive")
        return True

    visible_docs = []
    visible_metas = []
    visible_ids = []
    visible_dists = []

    filtered_count = 0
    public_summaries: List[str] = []
    filtered_details: List[Dict[str, Any]] = []

    for doc, meta, id_, dist in zip(raw_docs, raw_metas, raw_ids, raw_dists):
        try:
            if _allowed(meta, requester):
                visible_docs.append(doc)
                visible_metas.append(meta)
                visible_ids.append(id_)
                visible_dists.append(dist)
            else:
                filtered_count += 1
                if isinstance(meta, dict):
                    ps = meta.get("public_summary")
                    if ps and isinstance(ps, str) and ps.strip():
                        public_summaries.append(ps.strip())
                    filtered_details.append({
                        "id": id_,
                        "sensitivity": meta.get("sensitivity"),
                        "department": meta.get("department"),
                        "source": meta.get("source")
                    })
        except Exception as e:
            logger.exception("RBAC check failed for id=%s: %s", id_, e)
            continue

    context_text = "\n\n---\n\n".join(d for d in visible_docs if d)

    out: Dict[str, Any] = {
        "documents": visible_docs,
        "metadatas": visible_metas,
        "ids": visible_ids,
        "distances": visible_dists,
        "raw_documents": raw_docs,
        "raw_metadatas": raw_metas,
        "raw_ids": raw_ids,
        "context": context_text,
        "filtered_out_count": filtered_count,
        "public_summaries": public_summaries,
        "filtered_details": filtered_details,
    }

    # simple optional LLM over visible context
    if use_llm:
        global _llm_instance
        if _llm_instance is None:
            if LlamaCpp is not None:
                model_path = None
                for ext in ("*.gguf", "*.ggml", "*.bin"):
                    files = list(MODELS_DIR.glob(ext))
                    if files:
                        model_path = str(files[0])
                        break
                if model_path is None:
                    raise RuntimeError("No local LLM model found under models/. Provide model or disable use_llm.")
                _llm_instance = LlamaCpp(model_path=model_path, n_ctx=1024, n_batch=8, n_gpu_layers=0)
            else:
                raise RuntimeError("LLM not available. Install llama-cpp-python or pass an llm instance.")

        prompt = f"You are an internal assistant. Use the context below to answer concisely.\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{query_text}\n\nAnswer:"
        try:
            answer = _llm_instance(prompt, max_tokens=max_tokens, temperature=0.0)
        except TypeError:
            gen = _llm_instance.generate([prompt], max_new_tokens=max_tokens, temperature=0.0)
            answer = str(gen)
        out["answer"] = answer

    return out


def seed_manual_from_file(file_path: Optional[str] = None, source_name: Optional[str] = None) -> List[str]:
    from app.services.utility import get_data_path
    default_path = get_data_path("mission.txt")
    path = Path(file_path) if file_path else default_path
    if not path.exists():
        logger.warning("Seed file not found at %s", path)
        return []
    text = path.read_text(encoding="utf-8")
    return add_document_manual(source_name=source_name or path.name, text=text, metadata={"seeded": True})


def update_metadata_manual(ids: List[str], metadata: Dict[str, Any]) -> bool:
    client, collection = ensure_chroma_client(persist_directory=str(DEFAULT_PERSIST_DIR), collection_name=DEFAULT_COLLECTION_NAME)
    sanitized = sanitize_metadata_dict(metadata)
    return update_metadatas(collection=collection, ids=ids, metadata=sanitized)


def clear_collection_manual() -> None:
    client, collection = ensure_chroma_client(persist_directory=str(DEFAULT_PERSIST_DIR), collection_name=DEFAULT_COLLECTION_NAME)
    delete_all_documents(collection=collection)
