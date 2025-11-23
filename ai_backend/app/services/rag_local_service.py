from __future__ import annotations
import logging
import uuid
from pathlib import Path
# new import to fetch recent messages (tone is stored there by support_chat)
from app.services.support_chat import fetch_recent_messages
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

# Import centralized utilities
from app.services.utility import (
    BASE_DIR,
    DATA_DIR,
    DEFAULT_PERSIST_DIR,
    DEFAULT_COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
    get_local_embedding_model_path,
    get_embedding_model_instance,
    embed_texts,
    chunk_text_basic,
    sanitize_metadata_dict,
    build_tone_guidance,
    MODELS_DIR,
    get_data_path,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Model selection configuration
ENABLE_DYNAMIC_MODEL_SELECTION = False  # Set to True to enable dynamic model selection based on task
DEFAULT_MODEL_NAME = "mistral-7b-instruct-v0.2.Q3_K_M.gguf"  # Primary model to use

# Internal global handles - model cache
_llm_instances = {}  # Dict[str, Any] - cache for different model keys

# ---------- Utilities ----------

# ---------------------------
# MODEL ROUTING / SELECTION
# ---------------------------
def choose_model_for_task(task: str) -> str:
    """
    Choose appropriate model for a given task.
    
    Returns:
        "tiny" - for short chit-chat, quick responses
        "small" - for summarization, classification, tagging, intent detection
        "mistral" - for full RAG reasoning (default)
    
    Args:
        task: Task type - "chat", "summarize", "classify", "tag", "intent", "reason", "rag", etc.
    """
    task_lower = task.lower()
    
    # Small model tasks
    if task_lower in ["summarize", "classification", "classify", "tag", "tagging", "intent", "intent_detection"]:
        return "small"
    
    # Tiny model tasks
    if task_lower in ["chat", "chit-chat", "quick", "simple"]:
        return "tiny"
    
    # Default to mistral for RAG, reasoning, and unknown tasks
    return "mistral"


def get_llm_instance(model_key: str = "default"):
    """
    Lazy-load and cache LLM instances.
    
    By default, uses the specific model: mistral-7b-instruct-v0.2.Q3_K_M.gguf
    If ENABLE_DYNAMIC_MODEL_SELECTION is True and default model not found,
    falls back to dynamic selection based on model_key patterns.
    
    Args:
        model_key: Model identifier (only used if dynamic selection enabled)
                  "mistral" - Full RAG model
                  "small" - Smaller model for summarization/classification
                  "tiny" - Smallest model for quick chat
    
    Returns:
        LlamaCpp instance (cached)
    """
    global _llm_instances
    
    # Check cache first (use "default" as cache key for primary model)
    cache_key = "default"
    if model_key in _llm_instances:
        return _llm_instances[model_key]
    if cache_key in _llm_instances:
        return _llm_instances[cache_key]
    
    if LlamaCpp is None:
        raise RuntimeError("llama-cpp-python not installed. Install llama-cpp-python to use local LLM.")
    
    model_path = None
    config = {"n_ctx": 2048, "n_batch": 8}  # Default config for mistral
    
    # First, try to find the specific default model
    default_model_path = MODELS_DIR / DEFAULT_MODEL_NAME
    if default_model_path.exists():
        model_path = str(default_model_path)
        logger.info("Found default model: %s", model_path)
    else:
        # Try with different extensions
        for ext in [".gguf", ".ggml", ".bin"]:
            test_path = MODELS_DIR / (DEFAULT_MODEL_NAME.rsplit(".", 1)[0] + ext)
            if test_path.exists():
                model_path = str(test_path)
                logger.info("Found default model (with %s extension): %s", ext, model_path)
                break
    
    # If default model not found AND dynamic selection is enabled, do dynamic search
    if not model_path and ENABLE_DYNAMIC_MODEL_SELECTION:
        logger.info("Default model not found. Dynamic selection enabled. Searching by model_key='%s'", model_key)
        
        # Model file patterns by key
        model_patterns = {
            "mistral": ["*mistral*.gguf", "*mistral*.ggml", "*mistral*.bin"],
            "small": ["*small*.gguf", "*small*.ggml", "*small*.bin", "*7b*.gguf", "*7b*.ggml"],
            "tiny": ["*tiny*.gguf", "*tiny*.ggml", "*tiny*.bin", "*1b*.gguf", "*1b*.ggml", "*3b*.gguf", "*3b*.ggml"],
        }
        
        # Model configs by key (n_ctx, n_batch)
        model_configs = {
            "mistral": {"n_ctx": 2048, "n_batch": 8},
            "small": {"n_ctx": 1024, "n_batch": 4},
            "tiny": {"n_ctx": 512, "n_batch": 2},
        }
        
        patterns = model_patterns.get(model_key, model_patterns["mistral"])
        config = model_configs.get(model_key, model_configs["mistral"])
        
        # Search for model file using patterns
        for pattern in patterns:
            files = list(MODELS_DIR.glob(pattern))
            if files:
                model_path = str(files[0])
                logger.info("Found model via dynamic search: %s (pattern: %s)", model_path, pattern)
                break
        
        # Last resort: find any model file
        if not model_path:
            for ext in ("*.gguf", "*.ggml", "*.bin"):
                files = list(MODELS_DIR.glob(ext))
                if files:
                    model_path = str(files[0])
                    logger.warning("Using fallback model via dynamic search: %s", model_path)
                    break
    
    if not model_path:
        if ENABLE_DYNAMIC_MODEL_SELECTION:
            raise RuntimeError(
                f"No model file found in {MODELS_DIR}. "
                f"Expected '{DEFAULT_MODEL_NAME}' or dynamic selection patterns."
            )
        else:
            raise RuntimeError(
                f"Default model '{DEFAULT_MODEL_NAME}' not found in {MODELS_DIR}. "
                f"Set ENABLE_DYNAMIC_MODEL_SELECTION=True to enable dynamic model selection."
            )
    
    logger.info("Loading LlamaCpp model: path=%s, n_ctx=%d, n_batch=%d", model_path, config["n_ctx"], config["n_batch"])
    instance = LlamaCpp(
        model_path=model_path,
        n_ctx=config["n_ctx"],
        n_batch=config["n_batch"],
        n_gpu_layers=0  # CPU-only
    )
    
    # Cache the instance (use "default" as key for primary model)
    _llm_instances[cache_key] = instance
    return instance


# ---------------------------
# CHUNK-AWARE TOKEN BUDGET HELPERS
# ---------------------------


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

    selected_text = "\n\n---\n\n".join([(c.get(chunk_text_key) if isinstance(c, dict) else str(c)) for c in selected])
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
            top_selected_text = "\n\n---\n\n".join(
                [(c.get(chunk_text_key) if isinstance(c, dict) else str(c)) for c in top_selected])
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


# build_tone_guidance and sanitize_metadata_dict are now imported from utility.py

def inject_tone_into_prefix(prefix_text: str, tone: Optional[str]) -> str:
    """
    Inject a short 'Conversation Tone Guidance' block into an existing prefix.
    Keeps the original prefix but places the guidance near the top for clarity.
    """
    guidance = build_tone_guidance(tone)
    # Look for a natural insertion point: after the first newline or after a 'User Profile' section.
    # Simpler: place guidance at the beginning of the prefix so model sees it early.
    injected = (
        f"Conversation Tone Guidance:\n{guidance}\n\n"
        f"{prefix_text}"
    )
    return injected


# get_embedding_model_instance and embed_texts are now imported from utility.py


# chunk_text_basic is now imported from utility.py


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
    Note: LLM instances are now managed via get_llm_instance() with model routing.
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
        chunks = chunk_text_basic(text)

    if not chunks:
        logger.warning("No chunks produced for document: %s", source_name)
        return []

    # sanitize metadata and ensure source is present
    base_meta = metadata or {}
    sanitized_base = sanitize_metadata_dict(base_meta)
    sanitized_base["source"] = source_name
    # add ingestion timestamp if not present
    if "ingested_at" not in sanitized_base:
        from datetime import datetime
        sanitized_base["ingested_at"] = datetime.utcnow().isoformat() + "Z"

    metadatas = [dict(sanitized_base) for _ in chunks]
    ids = _generate_ids(prefix=source_name, n=len(chunks))

    # compute embeddings locally
    try:
        embeddings = embed_texts(chunks)
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


def query_local_rag(
        query_text: str,
        n_results: int = 3,
        requester: Optional[Dict[str, str]] = None,
        llm_prompt_prefix: Optional[str] = None,
        use_llm: bool = True,
        max_tokens: int = 256,
        session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query the local RAG:
    - compute local embedding for the query
    - retrieve top-k docs from Chroma
    - apply RBAC filtering
    - inject tone-aware guidance into LLM prefix
    """
    # Ensure Chroma client
    client, collection = ensure_chroma_client(
        persist_directory=str(DEFAULT_PERSIST_DIR),
        collection_name=DEFAULT_COLLECTION_NAME
    )

    if not query_text:
        raise ValueError("query_text must be provided")

    # -----------------------------
    # 1. Get embedding for query
    # -----------------------------
    try:
        q_emb = embed_texts([query_text])[0]
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

    # ------------------------------------------
    # 3. RBAC filtering (visible vs filtered)
    # ------------------------------------------
    def _allowed_by_metadata(meta: Optional[Dict[str, Any]], requester: Optional[Dict[str, str]]) -> bool:
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

        prompt = (
            f"{final_prefix}\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            f"QUESTION:\n{query_text}\n\nAnswer concisely:"
        )

        try:
            answer = _call_llm_with_retry(
                llm_instance,
                prompt,
                max_tokens=max_tokens,
                temperature=0.0
            )
        except Exception as e:
            logger.exception("LLM call failed: %s", e)
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
    default_path = get_data_path("company_overview.txt")
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
                    ids = add_document_to_rag_local(source_name=src_name, text=text, chunks=None,
                                                    metadata={"seeded": True})
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
