from __future__ import annotations
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
# Attempt imports (be explicit so runtime errors are clear)
try:
    import chromadb
    from chromadb.config import Settings
except Exception as e:
    chromadb = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    from langchain.llms import LlamaCpp
except Exception:
    LlamaCpp = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ---------- Configuration ----------
BASE_DIR = Path(__file__).resolve().parent.parent  # app/
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_PERSIST_DIR = BASE_DIR / "chroma_storage"
DEFAULT_COLLECTION_NAME = "local_manual_rag"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small & CPU-friendly

# Internal global handles
_chroma_client = None
_collection = None
_embedding_model = None
_llm_instance = None

# ---------- Utilities ----------

def _sanitize_meta_value(val):
    """
    Ensure metadata values are primitives (str/int/float/bool) for Chroma.
    - list of primitives -> comma-separated string
    - dict -> JSON string
    - others -> str()
    """
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
    # fallback
    return str(val)

def _sanitize_metadata_dict(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not meta:
        return {}
    return {str(k): _sanitize_meta_value(v) for k, v in meta.items()}

# small helper to update metadata for existing ids in chroma
def update_metadata(ids: List[str], metadata: Dict[str, Any]) -> bool:
    """
    Update metadata for existing document chunk ids.
    Returns True on success.
    Note: metadata will be sanitized so all values are primitives/strings.
    """
    if chromadb is None:
        raise ImportError("chromadb not installed")
    _, collection = _ensure_chroma_client()
    sanitized = _sanitize_metadata_dict(metadata)
    # Build per-id metadatas list (apply same metadata to each id)
    try:
        per_id_metas = [sanitized.copy() for _ in ids]
        # Many chroma clients support collection.update(ids=..., metadatas=...)
        # If update() not available, fallback to delete+re-add is not implemented here.
        collection.update(ids=ids, metadatas=per_id_metas)
        logger.info("Updated metadata for %d ids", len(ids))
        return True
    except Exception as e:
        logger.exception("Failed to update metadata for ids: %s", e)
        # Some chroma versions don't have update(); try alternative approaches
        try:
            # attempt to get full item and re-add with updated metadata
            # Note: this may not always be supported — keep as last-resort placeholder
            logger.debug("Attempting fallback metadata replacement for %d ids", len(ids))
            for i, _id in enumerate(ids):
                try:
                    # get existing doc (best-effort)
                    got = collection.get(ids=[_id])
                    docs = (got.get("documents") or [[]])[0]
                    metas = (got.get("metadatas") or [[]])[0]
                    if docs:
                        doc_text = docs[0]
                        old_meta = metas[0] if metas else {}
                        new_meta = {**old_meta, **sanitized}
                        collection.add(documents=[doc_text], ids=[_id], metadatas=[new_meta])
                except Exception:
                    continue
            logger.info("Fallback metadata update attempted.")
            return True
        except Exception:
            logger.exception("Fallback metadata update failed.")
            return False

def _get_local_embedding_model_path() -> Path:
    """
    If you want to place embedding models locally, put them under: <project_root>/embeddings_models/<EMBEDDING_MODEL_NAME>/
    This function returns that path.
    """
    return BASE_DIR.parent / "embeddings_models" / EMBEDDING_MODEL_NAME

def _ensure_chroma_client(persist_directory: Optional[str] = None):
    global _chroma_client, _collection
    if chromadb is None:
        raise ImportError("chromadb package is not installed. Install chromadb to use the local RAG.")
    if _chroma_client is None:
        persist_directory = persist_directory or str(DEFAULT_PERSIST_DIR)
        logger.info("Initializing Chroma persistent client at %s", persist_directory)
        # Try common init patterns to be resilient across chromadb versions
        try:
            # new API style
            from chromadb.config import Settings as _Settings
            _chroma_client = chromadb.Client(_Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))
        except Exception:
            try:
                # fallback older style
                _chroma_client = chromadb.PersistentClient(path=str(persist_directory))
            except Exception as e:
                logger.exception("Failed to initialize chroma client: %s", e)
                raise
    if _collection is None:
        try:
            _collection = _chroma_client.get_or_create_collection(name=DEFAULT_COLLECTION_NAME)
        except Exception:
            # fallback: attempt without options
            _collection = _chroma_client.get_or_create_collection(name=DEFAULT_COLLECTION_NAME)
    return _chroma_client, _collection

def _get_embedding_model_instance() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is not None:
        return _embedding_model
    if SentenceTransformer is None:
        raise ImportError("sentence_transformers not installed. Install sentence-transformers to compute local embeddings.")
    # prefer local cache first
    local_path = _get_local_embedding_model_path()
    if local_path.exists():
        logger.info("Loading embedding model from local path: %s", local_path)
        _embedding_model = SentenceTransformer(str(local_path))
    else:
        logger.info("Loading embedding model by name (may download if not cached): %s", EMBEDDING_MODEL_NAME)
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model

def _embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_embedding_model_instance()
    # convert_to_numpy True then .tolist() keeps persistence-friendly Python lists
    vectors = model.encode(texts, convert_to_numpy=True).tolist()
    return vectors

def _chunk_text_basic(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Produce overlapping chunks of the input text.
    Fixed so we always make progress and produce expected overlaps.
    """
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
        # advance start keeping overlap, but ensure progress by at least 1
        start = max(end - overlap, start + 1)
    return chunks

def _generate_ids(prefix: str, n: int) -> List[str]:
    return [f"{prefix}_{uuid.uuid4().hex}" for _ in range(n)]

# ---------- Public API ----------

def initialize_local_rag(embedding_model_instance: Optional[Any] = None,
                         llm_instance: Optional[Any] = None,
                         persist_directory: Optional[str] = None,
                         collection_name: Optional[str] = None) -> None:
    """
    Initialize resources:
    - Chroma persistent client / collection
    - (Optional) embedding_model_instance: if provided, will be used; else we create sentence-transformers locally
    - (Optional) llm_instance: if provided, will be used; else you can use a LlamaCpp instance created elsewhere

    After this call:
    - collection is ready (and seeded if empty depending on your usage)
    """
    global _embedding_model, _llm_instance, DEFAULT_COLLECTION_NAME, _chroma_client, _collection

    if collection_name:
        DEFAULT_COLLECTION_NAME = collection_name

    # embedding model assignment
    if embedding_model_instance is not None:
        _embedding_model = embedding_model_instance
        logger.info("Using provided embedding_model_instance")
    else:
        # lazy-load when needed via _get_embedding_model_instance()
        logger.info("No embedding_model_instance provided; will use local SentenceTransformer on demand")

    # LLM instance assignment
    if llm_instance is not None:
        _llm_instance = llm_instance
        logger.info("Using provided LLM instance")
    else:
        logger.info("No LLM instance provided; you may pass one later or rely on a separate local LLM service")

    # initialize chroma client & collection
    _ensure_chroma_client(persist_directory=persist_directory)
    logger.info("Local RAG initialized. Collection name: %s", DEFAULT_COLLECTION_NAME)


def add_document_to_rag_local(source_name: str,
                              text: str,
                              chunks: Optional[List[str]] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Add a document (or precomputed chunks) to the local chroma collection.

    Returns the list of ids added.

    Behavior:
    - Splits text into chunks if chunks not provided.
    - Computes embeddings locally for each chunk.
    - Sanitizes metadata to primitive types (strings/numbers/bools) for Chroma.
    - Adds documents, metadatas, ids, and embeddings to Chroma.
    - Returns the list of created chunk ids (explicit).
    """
    if chromadb is None:
        raise ImportError("chromadb not installed")

    _, collection = _ensure_chroma_client()

    # If client provided pre-chunked content, use it; otherwise chunk the text
    if not chunks:
        chunks = _chunk_text_basic(text)

    if not chunks:
        logger.warning("No chunks produced for document: %s", source_name)
        return []

    # default metadata applied per-chunk (sanitize so Chroma accepts values)
    base_meta = metadata or {}
    sanitized_base = _sanitize_metadata_dict(base_meta)
    sanitized_base["source"] = source_name
    # ensure ingested_at present if not provided
    if "ingested_at" not in sanitized_base:
        from datetime import datetime
        sanitized_base["ingested_at"] = datetime.utcnow().isoformat() + "Z"
    metadatas = [dict(sanitized_base) for _ in chunks]

    ids = _generate_ids(prefix=source_name, n=len(chunks))

    # compute embeddings locally
    try:
        embeddings = _embed_texts(chunks)
    except Exception as e:
        logger.exception("Failed to compute embeddings locally: %s", e)
        raise

    # Add to chroma
    try:
        collection.add(documents=chunks, metadatas=metadatas, ids=ids, embeddings=embeddings)
        logger.info("Added %d chunks for source %s to collection %s", len(chunks), source_name, DEFAULT_COLLECTION_NAME)
    except Exception as e:
        logger.exception("Failed to add documents to Chroma collection: %s", e)
        raise

    # Explicit return of ids (list of created chunk ids)
    return ids



def query_local_rag(query_text: str, n_results: int = 3, requester: Optional[Dict[str, str]] = None,
                    llm_prompt_prefix: Optional[str] = None, use_llm: bool = True, max_tokens: int = 256) -> Dict[str, Any]:
    """
    Query the local RAG:
    - compute local embedding for the query
    - retrieve top-k docs from Chroma
    - apply RBAC filtering (visible vs filtered)
    - returns both visible results and raw (pre-filter) results so the API layer can decide UX
    """
    _, collection = _ensure_chroma_client()

    if not query_text:
        raise ValueError("query_text must be provided")

    # compute query embedding
    try:
        q_emb = _embed_texts([query_text])[0]
        logger.debug("Computed query embedding (len=%d)", len(q_emb) if hasattr(q_emb, "__len__") else 0)
    except Exception as e:
        logger.exception("Failed to compute query embedding: %s", e)
        raise

    # query chroma (attempt embedding query, fallback to text query)
    try:
        result = collection.query(query_embeddings=[q_emb], n_results=n_results)
    except Exception:
        result = collection.query(query_texts=[query_text], n_results=n_results)

    # normalize result shape (support dict or object responses)
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
            logger.exception("Unexpected Chroma result format: %s", e)
            raw_docs, raw_metadatas, raw_ids, raw_distances = [], [], [], []

    # ------- RBAC / access control filtering -------
    def _allowed_by_metadata(meta: Optional[Dict[str, Any]], requester: Optional[Dict[str, str]]) -> bool:
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

    filtered_docs = []
    filtered_metas = []
    filtered_ids = []
    filtered_distances = []

    # additional UX/debug info collected from filtered-out docs
    filtered_out_count = 0
    public_summaries: List[str] = []
    filtered_details: List[Dict[str, Any]] = []

    # iterate over raw (unfiltered) results and split into visible vs filtered
    for doc, meta, id_, dist in zip(raw_docs, raw_metadatas, raw_ids, raw_distances):
        try:
            if _allowed_by_metadata(meta, requester):
                filtered_docs.append(doc)
                filtered_metas.append(meta)
                filtered_ids.append(id_)
                filtered_distances.append(dist)
            else:
                filtered_out_count += 1
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
                logger.debug("Filtered out document id=%s due to RBAC; requester=%s meta=%s", id_, requester, meta)
        except Exception as e:
            logger.exception("Error checking metadata access for id=%s: %s", id_, e)
            continue

    # build context text from allowed (visible) docs
    context_text = "\n\n---\n\n".join(d for d in filtered_docs if d)

    out = {
        # visible to requester after RBAC
        "documents": filtered_docs,
        "metadatas": filtered_metas,
        "ids": filtered_ids,
        "distances": filtered_distances,

        # raw (pre-filter) results so API layer can make UX decisions
        "raw_documents": raw_docs,
        "raw_metadatas": raw_metadatas,
        "raw_ids": raw_ids,
        "raw_distances": raw_distances,

        # UX and debug info
        "context": context_text,
        "filtered_out_count": filtered_out_count,
        "public_summaries": public_summaries,
        "filtered_details": filtered_details,
    }

    # Optionally call LLM on constructed prompt (only over visible docs/context)
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
                    raise RuntimeError("No LLM instance provided and no local GGUF model found under models/. Provide llm_instance when calling initialize_local_rag or set model file in models/.")
                try:
                    logger.info("Lazy-loading LlamaCpp model from %s", model_path)
                    _llm_instance = LlamaCpp(model_path=model_path, n_ctx=1024, n_batch=8, n_gpu_layers=0)
                except Exception as e:
                    logger.exception("Failed to initialize local LlamaCpp instance: %s", e)
                    raise RuntimeError("Failed to initialize local LLM") from e
            else:
                raise RuntimeError("No LLM instance available. Pass llm_instance to initialize_local_rag or install llama-cpp-python and place a model under models/")

        system_instructions = llm_prompt_prefix or (
            "You are a helpful assistant. Use the provided context to answer the question. "
            "If the answer is not present in the context, say you don't know."
        )
        prompt = f"{system_instructions}\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{query_text}\n\nAnswer concisely:"
        logger.debug("Prompt for LLM (trimmed): %s", prompt[:1000])

        try:
            answer = _llm_instance(prompt, max_tokens=max_tokens, temperature=0.0)
        except TypeError:
            try:
                gen = _llm_instance.generate([prompt], max_new_tokens=max_tokens, temperature=0.0)
                answer = str(gen)
            except Exception as e:
                logger.exception("Failed to generate with local LLM: %s", e)
                raise
        except Exception as e:
            logger.exception("LLM invocation failed: %s", e)
            raise

        out["answer"] = answer

    return out



def seed_from_file(file_path: Optional[str] = None, source_name: Optional[str] = None) -> List[str]:
    """
    Read the given file and index it. If file_path is None, attempts to seed from
    the default project data/mission.txt.
    Returns list of ids added.
    """
    # choose default mission path under project data/
    default_path = (BASE_DIR.parent / "data" / "mission.txt")
    path = Path(file_path) if file_path else default_path
    if not path.exists():
        logger.warning("Seed file not found at %s", path)
        return []

    text = path.read_text(encoding="utf-8")
    name = source_name or path.name
    return add_document_to_rag_local(source_name=name, text=text, chunks=None, metadata={"seeded": True})


def clear_collection() -> None:
    """
    Delete all documents from the collection. Use with caution.
    """
    _, collection = _ensure_chroma_client()
    try:
        # many chroma client APIs support delete with no args or purge
        # We attempt a few variants to remain compatible across versions.
        try:
            collection.delete()  # delete everything (if supported)
            logger.info("Cleared chroma collection using collection.delete()")
            return
        except Exception:
            pass

        # fallback: list ids and delete them
        all_ids = []
        try:
            # attempt to get ids via collection.get()
            coll_data = collection.get()
            # structure may vary: coll_data.get("ids") etc.
            if isinstance(coll_data, dict) and coll_data.get("ids"):
                # flatten and remove
                for ids_list in coll_data.get("ids", []):
                    all_ids.extend(ids_list)
        except Exception:
            pass

        # If we could not get IDs, attempt to drop & recreate collection via client (if available)
        try:
            client, _ = _ensure_chroma_client()
            client.delete_collection(name=DEFAULT_COLLECTION_NAME)
            # re-create
            _ensure_chroma_client()
            logger.info("Deleted and recreated collection %s via client.delete_collection()", DEFAULT_COLLECTION_NAME)
            return
        except Exception:
            pass

        # last resort: if we have ids, delete by ids
        if all_ids:
            collection.delete(ids=all_ids)
            logger.info("Cleared chroma collection by deleting %d ids", len(all_ids))
            return

        logger.warning("Unable to clear collection using available APIs; collection may still contain documents.")
    except Exception as e:
        logger.exception("Error clearing collection: %s", e)
        raise
