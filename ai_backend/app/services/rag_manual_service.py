from __future__ import annotations
import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

# Embedding import (optional at runtime)
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

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

BASE_DIR = Path(__file__).resolve().parent.parent  # app/
DEFAULT_PERSIST_DIR = BASE_DIR / "chroma_storage"
DEFAULT_COLLECTION_NAME = "local_manual_rag"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

_embedding_model = None
_llm_instance = None


# -------------------
# Utilities (small, self-contained)
# -------------------
def _get_local_embedding_model_path() -> Path:
    return BASE_DIR.parent / "embeddings_models" / EMBEDDING_MODEL_NAME

def _get_embedding_model_instance():
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    if SentenceTransformer is None:
        raise ImportError("sentence_transformers is required for local embeddings.")
    local_path = _get_local_embedding_model_path()
    if local_path.exists():
        logger.info("Loading embedding model from local path: %s", local_path)
        _embedding_model = SentenceTransformer(str(local_path))
    else:
        logger.info("Loading embedding model by name: %s", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model

def _embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_embedding_model_instance()
    vectors = model.encode(texts, convert_to_numpy=True).tolist()
    return vectors

def _chunk_text_basic(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        if end == L:
            break
        start = max(end - overlap, start + 1)
    return chunks

def _sanitize_meta_value(val):
    import json
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, list):
        if all(isinstance(x, (str, int, float, bool)) for x in val):
            return ",".join(str(x) for x in val)
        return json.dumps(val, ensure_ascii=False)
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    return str(val)

def _sanitize_metadata_dict(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not meta:
        return {}
    return {str(k): _sanitize_meta_value(v) for k, v in meta.items()}


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

    chunks = _chunk_text_basic(text)
    if not chunks:
        logger.warning("No chunks produced for %s", source_name)
        return []

    base_meta = metadata or {}
    sanitized_base = _sanitize_metadata_dict(base_meta)
    sanitized_base["source"] = source_name
    if "ingested_at" not in sanitized_base:
        from datetime import datetime
        sanitized_base["ingested_at"] = datetime.utcnow().isoformat() + "Z"

    metadatas = [dict(sanitized_base) for _ in chunks]
    ids = [f"{source_name}_{uuid.uuid4().hex}" for _ in chunks]

    try:
        embeddings = _embed_texts(chunks)
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
    q_emb = _embed_texts([query_text])[0]

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
                models_dir = BASE_DIR.parent / "models"
                model_path = None
                for ext in ("*.gguf", "*.ggml", "*.bin"):
                    files = list(models_dir.glob(ext))
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
    default_path = BASE_DIR.parent / "data" / "mission.txt"
    path = Path(file_path) if file_path else default_path
    if not path.exists():
        logger.warning("Seed file not found at %s", path)
        return []
    text = path.read_text(encoding="utf-8")
    return add_document_manual(source_name=source_name or path.name, text=text, metadata={"seeded": True})


def update_metadata_manual(ids: List[str], metadata: Dict[str, Any]) -> bool:
    client, collection = ensure_chroma_client(persist_directory=str(DEFAULT_PERSIST_DIR), collection_name=DEFAULT_COLLECTION_NAME)
    sanitized = _sanitize_metadata_dict(metadata)
    return update_metadatas(collection=collection, ids=ids, metadata=sanitized)


def clear_collection_manual() -> None:
    client, collection = ensure_chroma_client(persist_directory=str(DEFAULT_PERSIST_DIR), collection_name=DEFAULT_COLLECTION_NAME)
    delete_all_documents(collection=collection)
