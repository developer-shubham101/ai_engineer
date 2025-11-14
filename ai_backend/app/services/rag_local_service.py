"""
rag_local_service.py

A drop-in local version of your previous rag_manual_service.py.

Provides:
- initialize_local_rag(embedding_model_instance=None, llm_instance=None)
- add_document_to_rag_local(source_name: str, text: str, chunks: Optional[List[str]] = None, metadata: Optional[dict] = None)
- query_local_rag(query_text: str, n_results: int = 3)

Dependencies:
- chromadb
- sentence_transformers (all-MiniLM-L6-v2 recommended)
- langchain (LlamaCpp wrapper) + llama-cpp-python installed and GGUF/ggml model present locally

Notes:
- This file intentionally uses local embeddings + local LLM only.
- It is named differently from your google-backed service so you can keep both.
"""

from __future__ import annotations
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any
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
        _chroma_client = chromadb.PersistentClient(path=str(persist_directory))
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
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        start = max(end - overlap, end)
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

    - Splits text into chunks if chunks not provided.
    - Computes embeddings locally for each chunk.
    - Adds documents, metadatas, ids, and embeddings to Chroma.
    """
    if chromadb is None:
        raise ImportError("chromadb not installed")

    _, collection = _ensure_chroma_client()

    if not chunks:
        chunks = _chunk_text_basic(text)

    if not chunks:
        logger.warning("No chunks produced for document: %s", source_name)
        return []

    metadatas = [{**(metadata or {}), "source": source_name} for _ in chunks]
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

    return ids


def query_local_rag(query_text: str, n_results: int = 3, llm_prompt_prefix: Optional[str] = None,
                    use_llm: bool = True, max_tokens: int = 256) -> Dict[str, Any]:
    """
    Query the local RAG:
    - compute local embedding for the query
    - retrieve top-k docs from Chroma
    - if use_llm True: build prompt and call local LLM instance (if available)
    - returns a dict containing retrieved docs, metadatas, ids, distances and optionally `answer`

    If no LLM instance is available and use_llm is True, raises RuntimeError.
    """
    _, collection = _ensure_chroma_client()

    if not query_text:
        raise ValueError("query_text must be provided")

    # compute query embedding
    try:
        q_emb = _embed_texts([query_text])[0]
        print("MARK:- q_emb", q_emb)
    except Exception as e:
        logger.exception("Failed to compute query embedding: %s", e)
        raise

    # query chroma
    try:
        result = collection.query(query_embeddings=[q_emb], n_results=n_results)
    except Exception:
        # fallback to text query if query_embeddings not supported by this client
        result = collection.query(query_texts=[query_text], n_results=n_results)

    # normalize result shape
    if isinstance(result, dict):
        docs = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
    else:
        # some chroma clients return objects; try attribute access
        try:
            docs = result.documents[0]
            metadatas = result.metadatas[0]
            ids = result.ids[0]
            distances = result.distances[0] if hasattr(result, "distances") else []
        except Exception as e:
            logger.exception("Unexpected Chroma result format: %s", e)
            docs = []
            metadatas = []
            ids = []
            distances = []

    # build context
    context_text = "\n\n---\n\n".join(d for d in docs if d)

    out = {
        "documents": docs,
        "metadatas": metadatas,
        "ids": ids,
        "distances": distances,
        "context": context_text,
    }

    # Optionally call LLM on constructed prompt
    if use_llm:
        # Ensure llm instance is available, prefer provided global _llm_instance
        global _llm_instance
        if _llm_instance is None:
            # attempt to lazy-load a LlamaCpp instance if LlamaCpp is available and an environment model is present
            if LlamaCpp is not None:
                # try to find a model file in project models/ directory
                models_dir = BASE_DIR.parent / "models"
                # pick the first gguf/ggml file we find
                print("models_dir", models_dir)
                model_path = None
                for ext in ("*.gguf", "*.ggml", "*.bin"):
                    files = list(models_dir.glob(ext))
                    print("files", files)
                    if files:
                        model_path = str(files[0])
                        break
                print("model_path", model_path)
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

        # Build prompt
        system_instructions = llm_prompt_prefix or (
            "You are a helpful assistant. Use the provided context to answer the question. "
            "If the answer is not present in the context, say you don't know."
        )
        prompt = f"{system_instructions}\n\nCONTEXT:\n{context_text}\n\nQUESTION:\n{query_text}\n\nAnswer concisely:"
        logger.debug("Prompt for LLM (trimmed): %s", prompt[:1000])

        # Call LLM - supports LangChain LlamaCpp wrapper which is callable
        try:
            # The LangChain LlamaCpp wrapper accepts (prompt, max_tokens=.., temperature=..)
            answer = _llm_instance(prompt, max_tokens=max_tokens, temperature=0.0)
        except TypeError:
            # fallback in case API differs
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
