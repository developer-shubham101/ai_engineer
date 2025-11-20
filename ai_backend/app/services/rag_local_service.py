from __future__ import annotations
import logging
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

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
    get_documents_by_ids,
    update_metadatas,
    delete_ids,
    delete_all_documents,
    delete_collection_by_name,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ---------- Configuration ----------
BASE_DIR = Path(__file__).resolve().parent.parent  # app/
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_PERSIST_DIR = BASE_DIR / "chroma_storage"
DEFAULT_COLLECTION_NAME = "local_manual_rag"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # small & CPU-friendly

# Internal global handles
_embedding_model = None
_llm_instance = None

# ---------- Utilities ----------

# ---------------------------
# CHUNK-AWARE TOKEN BUDGET HELPERS
# ---------------------------
import logging
logger = logging.getLogger(__name__)

def estimate_tokens_from_text(text: str, chars_per_token: float = 4.0) -> int:
    """
    Heuristic: approximate tokens from characters.
    chars_per_token default 4.0 (typical for English).
    Always return at least 1 token for non-empty text.
    """
    if not text:
        return 0
    # integer math (safe, precise)
    token_est = int(len(text) / chars_per_token)
    if token_est < 1:
        token_est = 1
    return token_est

def select_chunks_by_token_budget(
    chunks: list,
    prefix_text: str,
    question_text: str,
    n_ctx: int,
    requested_max_tokens: int,
    safety_margin: int = 32,
    chars_per_token: float = 4.0,
    chunk_text_key: str = "content",
    chunk_score_key: str = "score"
):
    """
    Select whole chunks (not partial) to fit inside the available token budget.

    Parameters
    - chunks: list of chunk dicts (expected to have text in chunk_text_key and optionally score in chunk_score_key)
              e.g. [{"content": ".....", "score": 0.92}, ...]
    - prefix_text: the prompt prefix (system + profile + history) that will appear before CONTEXT
    - question_text: the user's question text that will appear after CONTEXT
    - n_ctx: model context window (e.g., 2048)
    - requested_max_tokens: tokens you will ask the model to generate (e.g., 256)
    - safety_margin: reserve a few tokens to avoid edge cases
    - chars_per_token: heuristic chars per token
    - chunk_text_key: dict key that holds chunk text
    - chunk_score_key: dict key that holds retrieval score (higher = more relevant); if missing, original order used

    Returns:
    - selected_chunks: list of chunk dicts chosen (in descending score order)
    - selected_text: concatenated text of selected chunks
    - used_tokens_est: total estimated tokens for prefix + selected chunks + question
    - available_budget_tokens: token budget that was used/left
    """
    # estimate prefix + question token usage
    prefix_tokens = estimate_tokens_from_text(prefix_text, chars_per_token)
    question_tokens = estimate_tokens_from_text(question_text, chars_per_token)
    gen_tokens = int(requested_max_tokens)
    # compute available tokens for CONTEXT
    available_for_context = n_ctx - (prefix_tokens + question_tokens + gen_tokens + safety_margin)
    if available_for_context <= 0:
        # nothing can be added: context budget exhausted (prefix+question+gen too large).
        logger.warning(
            "Token budget for context <= 0 (available_for_context=%s). "
            "prefix_tokens=%s question_tokens=%s gen_tokens=%s safety_margin=%s n_ctx=%s",
            available_for_context, prefix_tokens, question_tokens, gen_tokens, safety_margin, n_ctx
        )
        return [], "", prefix_tokens + question_tokens + gen_tokens, available_for_context

    # sort chunks by score if available, else keep provided order
    try:
        sorted_chunks = sorted(
            chunks,
            key=lambda c: c.get(chunk_score_key, 0.0),
            reverse=True
        )
    except Exception:
        sorted_chunks = list(chunks)

    selected = []
    used_context_tokens = 0

    for ch in sorted_chunks:
        text = ch.get(chunk_text_key) if isinstance(ch, dict) else str(ch)
        if not text:
            continue
        tkns = estimate_tokens_from_text(text, chars_per_token)
        # if adding this chunk exceeds budget, skip it
        if used_context_tokens + tkns > available_for_context:
            logger.debug("Skipping a chunk: would exceed context budget (used=%s + chunk=%s > avail=%s)",
                         used_context_tokens, tkns, available_for_context)
            continue
        selected.append(ch)
        used_context_tokens += tkns

    selected_text = "\n\n---\n\n".join([ (c.get(chunk_text_key) if isinstance(c, dict) else str(c)) for c in selected ])
    total_used_est = prefix_tokens + used_context_tokens + question_tokens + gen_tokens
    return selected, selected_text, total_used_est, available_for_context - used_context_tokens

def build_prompt_with_selected_chunks(prefix: str, context_text: str, question: str) -> str:
    """
    Build a consistent prompt using markers that downstream retry/trim helpers can detect.
    """
    parts = []
    if prefix:
        parts.append(prefix.rstrip())
    parts.append("\n\nCONTEXT:\n")
    if context_text:
        parts.append(context_text.rstrip())
    else:
        parts.append("[NO_CONTEXT_AVAILABLE]")
    parts.append("\n\nQUESTION:\n")
    parts.append(question.rstrip())
    return "".join(parts)

# ---------------------------
# USAGE: integrate into query_local_rag
# ---------------------------
# Example snippet to replace the old prompt-building + direct llm call inside query_local_rag:
#
# (1) you must have:
#   - `retrieved_chunks` : list of retrieved chunk dicts (each with 'content' and optional 'score')
#   - `_llm_instance` : your initialized LlamaCpp/langchain wrapper
#   - `n_ctx` : the context window used when instantiating the model (e.g., 2048)
#   - `max_tokens` : requested generation tokens (e.g., 256)
#   - `prefix_text` : system prefix + profile + history
#   - `question_text` : the incoming user question
#
# (2) Replace the direct call:
#     answer = _llm_instance(prompt, max_tokens=max_tokens, temperature=0.0)
#
# with the following block:
#
def _invoke_llm_with_chunk_budget(
    llm_instance,
    retrieved_chunks,
    prefix_text,
    question_text,
    n_ctx=2048,
    max_tokens=256,
    safety_margin=32,
    chunk_text_key="content",
    chunk_score_key="score",
    chars_per_token=4.0
):
    """
    Wrapper to select chunks based on token budget, build prompt, and call the LLM.
    Returns the raw model output (string) and metadata about selection.
    """
    # 1) select chunks that fit in the budget
    selected_chunks, selected_text, used_est, remaining = select_chunks_by_token_budget(
        chunks=retrieved_chunks,
        prefix_text=prefix_text,
        question_text=question_text,
        n_ctx=n_ctx,
        requested_max_tokens=max_tokens,
        safety_margin=safety_margin,
        chars_per_token=chars_per_token,
        chunk_text_key=chunk_text_key,
        chunk_score_key=chunk_score_key
    )

    # 2) build prompt
    prompt = build_prompt_with_selected_chunks(prefix_text, selected_text, question_text)

    # 3) attempt LLM call (direct; you can substitute your retry wrapper here if you have one)
    try:
        logger.debug("Calling LLM with estimated total tokens=%s (remaining budget=%s). Selected chunks=%d",
                     used_est, remaining, len(selected_chunks))
        output = llm_instance(prompt, max_tokens=max_tokens, temperature=0.0)
    except ValueError as ve:
        # If the model still complains, attempt a fallback: shrink selected chunks count (drop lowest-scored half) and retry once.
        msg = str(ve)
        logger.warning("LLM raised ValueError on call: %s", msg)
        if ("exceed context window" in msg) or ("Requested tokens" in msg) or ("context window" in msg):
            # conservative retry: keep only top 50% of selected chunks
            if len(selected_chunks) <= 1:
                # nothing to drop; re-raise
                raise
            # determine how many to keep
            keep_count = max(1, int(len(selected_chunks) * 0.5))
            top_selected = selected_chunks[:keep_count]
            top_selected_text = "\n\n---\n\n".join([ (c.get(chunk_text_key) if isinstance(c, dict) else str(c)) for c in top_selected ])
            retry_prompt = build_prompt_with_selected_chunks(prefix_text, top_selected_text, question_text)
            logger.warning("Retrying LLM with fewer chunks (kept %d of %d)", keep_count, len(selected_chunks))
            return llm_instance(retry_prompt, max_tokens=max_tokens, temperature=0.0), {
                "selected_count": keep_count,
                "original_selected": len(selected_chunks),
                "retry": "dropped_low_half"
            }
        # else unknown ValueError -> re-raise
        raise

    # 4) return output + metadata
    meta = {
        "selected_count": len(selected_chunks),
        "original_selected": len(retrieved_chunks),
        "estimated_tokens_used": used_est,
        "remaining_context_tokens": remaining
    }
    return output, meta

# -------------------------------------------------------------------------
# Example: INTEGRATION POINT in query_local_rag (pseudo-placement)
# -------------------------------------------------------------------------
# Replace your old logic (where you build 'prompt' using all chunks and call _llm_instance)
# with code like this:

# prefix_text = build_prefix(...)  # whatever you already build: system + profile + history
# question_text = incoming_user_question
# retrieved_chunks = results_from_chroma  # ensure each chunk has 'content' and optionally 'score'

# Set these to your actual values:
# n_ctx should match what you used when instantiating LlamaCpp; e.g., 2048
# max_tokens is requested generation tokens from the API or default
#
# Example call:
# answer_text, selection_meta = _invoke_llm_with_chunk_budget(
#     _llm_instance,
#     retrieved_chunks,
#     prefix_text,
#     question_text,
#     n_ctx=2048,
#     max_tokens=max_tokens,
#     safety_margin=32,
#     chunk_text_key="content",
#     chunk_score_key="score",
#     chars_per_token=4.0
# )
#
# Now use answer_text as the model's output and log/use selection_meta for diagnostics.



def _get_local_embedding_model_path() -> Path:
    """
    If you want to place embedding models locally, put them under: <project_root>/embeddings_models/<EMBEDDING_MODEL_NAME>/
    This function returns that path.
    """
    return BASE_DIR.parent / "embeddings_models" / EMBEDDING_MODEL_NAME

def _sanitize_meta_value(val):
    """
    Ensure metadata values are primitives (str, int, float, bool) for Chroma.
    - If val is list of primitives -> join with commas
    - If val is dict -> json.dumps
    - Else convert to str
    """
    import json
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, list):
        # if list of primitives, join; otherwise json-dump
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
    - ensures Chroma client & collection
    - optionally set provided embedding and llm instances
    """
    global _embedding_model, _llm_instance

    if embedding_model_instance is not None:
        _embedding_model = embedding_model_instance
        logger.info("Using provided embedding_model_instance")
    else:
        logger.info("No embedding_model_instance provided; will lazy-load when needed")

    if llm_instance is not None:
        _llm_instance = llm_instance
        logger.info("Using provided LLM instance")
    else:
        logger.info("No LLM instance provided; local LLM may be lazy-loaded on demand")

    # Ensure Chroma client & collection exist
    ensure_chroma_client(persist_directory=str(persist_directory or DEFAULT_PERSIST_DIR),
                         collection_name=collection_name or DEFAULT_COLLECTION_NAME)
    logger.info("Local RAG initialization completed (collection: %s)", collection_name or DEFAULT_COLLECTION_NAME)


def add_document_to_rag_local(source_name: str,
                              text: str,
                              chunks: Optional[List[str]] = None,
                              metadata: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Add a document (or precomputed chunks) to the local chroma collection.

    Returns the list of ids added.

    - Splits text into chunks if chunks not provided.
    - Computes embeddings locally for each chunk.
    - Adds documents, metadatas, ids, and embeddings to Chroma via chroma_utils.
    """
    import json

    if not chunks:
        chunks = _chunk_text_basic(text)

    if not chunks:
        logger.warning("No chunks produced for document: %s", source_name)
        return []

    # sanitize metadata and ensure source is present
    base_meta = metadata or {}
    sanitized_base = _sanitize_metadata_dict(base_meta)
    sanitized_base["source"] = source_name
    # add ingestion timestamp if not present
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

    # Add to chroma via helper
    try:
        client, collection = ensure_chroma_client(persist_directory=str(DEFAULT_PERSIST_DIR), collection_name=DEFAULT_COLLECTION_NAME)
        add_documents_to_collection(collection=collection, documents=chunks, metadatas=metadatas, ids=ids, embeddings=embeddings)
        logger.info("Added %d chunks for source %s to collection %s", len(chunks), source_name, DEFAULT_COLLECTION_NAME)
    except Exception as e:
        logger.exception("Failed to add documents to Chroma collection: %s", e)
        raise

    return ids




def _call_llm_with_retry(
    llm_instance,
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    retry_shrink_ratio: float = 0.5,
    min_keep_chars: int = 200
):
    """
    Try calling LLM once. If ValueError says context too large:
    - Trim CONTEXT block to 50%
    - Retry once safely
    """

    try:
        # First attempt
        return llm_instance(prompt, max_tokens=max_tokens, temperature=temperature)

    except ValueError as ve:
        msg = str(ve)
        if ("exceed context window" in msg) or ("Requested tokens" in msg):
            logger.warning("Context window error. Retrying with trimmed context: %s", msg)

            # detect structure: <prefix> CONTEXT <ctx> QUESTION <rest>
            ctx_marker = "\n\nCONTEXT:\n"
            q_marker = "\n\nQUESTION:\n"

            try:
                # split into prefix, ctx, rest
                before, after = prompt.split(ctx_marker, 1)
                ctx_text, rest = after.split(q_marker, 1)

                # trim context size
                keep_len = max(min_keep_chars, int(len(ctx_text) * retry_shrink_ratio))
                head_keep = int(keep_len * 0.6)
                tail_keep = keep_len - head_keep

                new_ctx = ctx_text[:head_keep] + "\n...\n" + ctx_text[-tail_keep:]

                new_prompt = (
                    before + ctx_marker + new_ctx + q_marker + rest
                )

                logger.debug(
                    "Retrying LLM: Original ctx=%d, trimmed=%d",
                    len(ctx_text), len(new_ctx)
                )

                # Retry now
                return llm_instance(new_prompt, max_tokens=max_tokens, temperature=temperature)

            except Exception as e:
                logger.exception("Failed during trim retry: %s", e)
                raise ve  # give original ValueError

        # Not a token error → re-raise
        raise



def query_local_rag(query_text: str, n_results: int = 3, requester: Optional[Dict[str, str]] = None,
                    llm_prompt_prefix: Optional[str] = None, use_llm: bool = True, max_tokens: int = 256) -> Dict[str, Any]:
    """
    Query the local RAG:
    - compute local embedding for the query
    - retrieve top-k docs from Chroma (via chroma_utils)
    - apply RBAC filtering (visible vs filtered)
    - returns both visible results and raw (pre-filter) results so the API layer can decide UX
    """
    client, collection = ensure_chroma_client(persist_directory=str(DEFAULT_PERSIST_DIR), collection_name=DEFAULT_COLLECTION_NAME)

    if not query_text:
        raise ValueError("query_text must be provided")

    # compute query embedding
    try:
        q_emb = _embed_texts([query_text])[0]
        logger.debug("Computed query embedding (len=%d)", len(q_emb) if hasattr(q_emb, "__len__") else 0)
    except Exception as e:
        logger.exception("Failed to compute query embedding: %s", e)
        raise

    # query chroma via helper (embedding preferred, text fallback)
    try:
        result = query_collection(collection=collection, query_embeddings=[q_emb], n_results=n_results)
    except Exception:
        result = query_collection(collection=collection, query_texts=[query_text], n_results=n_results)

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
        """
        Simple RBAC rules. Expected meta keys:
          - sensitivity: one of public_internal | department_confidential | role_confidential | highly_confidential | personal
          - department: department string
          - allowed_roles: optional list of roles allowed
          - owner_id: for personal items
        requester keys expected: role, department, user_id
        """
        if not meta:
            return requester is not None  # if requester present, allow public_internal by default
        sens = meta.get("sensitivity", "public_internal")
        # personal: only owner or HR/Legal/Executive
        if sens == "personal":
            owner = meta.get("owner_id")
            if owner == (requester.get("user_id") if requester else None):
                return True
            return requester and requester.get("role") in ("HR", "Legal", "Executive")
        # highly_confidential: only Legal / Executive
        if sens == "highly_confidential":
            return requester and requester.get("role") in ("Legal", "Executive")
        # role_confidential: check allowed_roles
        if sens == "role_confidential":
            allowed = meta.get("allowed_roles") or []
            if requester and requester.get("role") in allowed:
                return True
            return requester and requester.get("role") in ("HR", "Legal", "Executive")
        # department_confidential: same department or HR/Legal/Executive
        if sens == "department_confidential":
            if requester and requester.get("department") == meta.get("department"):
                return True
            return requester and requester.get("role") in ("HR", "Legal", "Executive")
        # public_internal or unknown: allow
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

    out: Dict[str, Any] = {
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
                    _llm_instance = LlamaCpp(model_path=model_path, n_ctx=2048, n_batch=8, n_gpu_layers=0)
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
            answer = _call_llm_with_retry(
                _llm_instance,
                prompt,
                max_tokens=max_tokens,
                temperature=0.0
            )
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
    Read the given file or directory and index it.

    Behavior:
    - If file_path is None: attempts to seed from default project data/company_overview.txt.
    - If file_path is a file: read & ingest that single file.
    - If file_path is a directory: iterate non-recursively through files in the directory
      and ingest each file found (skip directories). Returns a flat list of all chunk ids added.

    Returns list of ids added (may be empty).
    """
    default_path = (BASE_DIR.parent / "data" / "company_overview.txt")
    path = Path(file_path) if file_path else default_path
    if not path.exists():
        logger.warning("Seed path not found at %s", path)
        return []

    added_ids: List[str] = []

    # If path is a directory, iterate files (non-recursive) and ingest each
    if path.is_dir():
        logger.info("Seeding directory: %s", path)
        for child in sorted(path.iterdir()):
            if child.is_file():
                try:
                    text = child.read_text(encoding="utf-8")
                    src_name = source_name or child.name
                    ids = add_document_to_rag_local(source_name=src_name, text=text, chunks=None, metadata={"seeded": True})
                    if ids:
                        added_ids.extend(ids)
                        logger.info("Seeded file %s -> %d chunks", child.name, len(ids))
                except Exception as e:
                    logger.exception("Failed to seed file %s: %s", child, e)
                    continue
        return added_ids

    # Otherwise, it's a single file; ingest it.
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.exception("Failed to read seed file %s: %s", path, e)
        return []

    name = source_name or path.name
    try:
        ids = add_document_to_rag_local(source_name=name, text=text, chunks=None, metadata={"seeded": True})
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
    client, collection = ensure_chroma_client(persist_directory=str(DEFAULT_PERSIST_DIR), collection_name=DEFAULT_COLLECTION_NAME)
    sanitized = _sanitize_metadata_dict(metadata)
    return update_metadatas(collection=collection, ids=ids, metadata=sanitized)


def clear_collection() -> None:
    """
    Delete all documents from the collection. Use with caution.
    """
    client, collection = ensure_chroma_client(persist_directory=str(DEFAULT_PERSIST_DIR), collection_name=DEFAULT_COLLECTION_NAME)
    try:
        delete_all_documents(collection=collection, client=client, collection_name=DEFAULT_COLLECTION_NAME)
    except Exception as e:
        logger.exception("Error clearing collection: %s", e)
        raise
