# app/api_routes_rag.py

from typing import List, Optional, Dict, Any
import logging
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Header
from pydantic import BaseModel, Field
from app.services.sentiment_classifier import get_global_sentiment
from app.services.support_chat import get_sentiment_stats

# Import dependency providers
from app.dependencies import get_rag_service, get_current_user, get_current_user_optional, require_roles
from app.services.user_service import get_all_user_meta, set_user_meta

from app.services.google_models import query_google_rag
from app.services import support_chat

from app.services.support_chat import (
    get_next_missing_profile_key,
    set_profile_value,
    get_full_profile,
)
from app.utils.doc_parser import parse_text, RawFormat
import os

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rag", tags=["RAG"])
support_chat.init_support_chat_db(reset_on_start=True)

# ---------------------------
# Models
# ---------------------------
class SentimentRequest(BaseModel):
    text: str

class RetrievedDoc(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None
    distance: Optional[float] = None

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    use_llm: bool = False
    max_tokens: int = 256
    category: Optional[str] = None

class QueryResponse(BaseModel):
    answer: Optional[str] = None
    retrieved: List[RetrievedDoc] = Field(default_factory=list)
    context: Optional[str] = None

class AddDocRequest(BaseModel):
    source_name: str
    text: str
    metadata: Optional[Dict[str, Any]] = None

class AddResponse(BaseModel):
    message: str
    chunk_count: int = 0


class SupportSessionStartRequest(BaseModel):
    session_id: Optional[str] = None
    name: Optional[str] = None
    sex: Optional[str] = None
    position: Optional[str] = None
    category: Optional[str] = None
    notes: Optional[str] = None


class SupportSessionStartResponse(BaseModel):
    session_id: str
    message: str


class SupportSessionEndRequest(BaseModel):
    session_id: str


class SupportSessionEndResponse(BaseModel):
    session_id: str
    message: str

# ---------------------------
# Helpers / Config
# ---------------------------

ALLOWED_SENSITIVITY = {
    "public_internal",
    "department_confidential",
    "role_confidential",
    "highly_confidential",
    "personal",
}


def validate_metadata(meta: Optional[Dict[str, Any]]):
    if not meta:
        return
    sens = meta.get("sensitivity")
    if sens and sens not in ALLOWED_SENSITIVITY:
        raise HTTPException(status_code=400, detail=f"Invalid sensitivity '{sens}'. Allowed: {list(ALLOWED_SENSITIVITY)}")




# ---------------------------
# Routes
# ---------------------------

@router.post("/{model_provider}/query", response_model=QueryResponse)
async def query(
    model_provider: str,
    req: QueryRequest,
    requester: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    rag_service=Depends(get_rag_service),
):
    """
    Query the RAG. Supports session-aware onboarding and personalization.
    This endpoint can be used with different model providers (e.g., 'local', 'google').
    """
    logger.info("Query request: provider=%s, role=%s user=%s question=%s", 
                model_provider, 
                requester.get("role") if requester else "Guest", 
                requester.get("user_id") if requester else None, 
                req.question)

    llm_prefix = None
    session_history = []
    session_id = requester.get("user_id") if requester else None  # Use user_id as session_id

    # For authenticated users, use their user_id as session
    if session_id:
        # Touch session to update last activity
        if support_chat.session_exists(session_id):
            try:
                support_chat.touch_session(
                    session_id=session_id,
                    role=requester.get("role"),
                    department=requester.get("department"),
                )
            except ValueError as exc:
                logger.warning(f"Session touch failed: {exc}")

        # Fetch conversation history
        session_history = support_chat.fetch_recent_messages(
            session_id=session_id,
            limit=support_chat.MAX_HISTORY_TURNS,
        )

    # Check user profile and handle onboarding
    if session_id:
        # Get profile from user_meta table
        profile = get_all_user_meta(session_id)
        next_field = get_next_missing_profile_key(session_id)

        # If user has missing profile fields, handle onboarding
        if next_field:
            last_assistant_msg = None
            if session_history:
                last_msg = session_history[-1]
                if last_msg.get("speaker", "").lower() == "assistant":
                    last_assistant_msg = last_msg.get("content", "")

            expected_question = next_field["question"]
            key_to_save = next_field["key"]

            # Check if this is a response to the onboarding question
            if last_assistant_msg and last_assistant_msg.strip() == expected_question.strip():
                user_reply = req.question.strip()

                try:
                    support_chat.store_message(session_id, "user", req.question)
                except Exception:
                    logger.exception("Failed to store user onboarding reply (non-fatal)")

                # Save to user_meta instead of session profile
                try:
                    set_user_meta(session_id, key_to_save, user_reply)
                except Exception as exc:
                    logger.exception("Failed to save onboarding value: %s", exc)
                    raise HTTPException(status_code=500, detail="Failed to save onboarding data.")

                # Check for next missing field
                next_field = get_next_missing_profile_key(session_id)
                if next_field:
                    try:
                        support_chat.store_message(session_id, "assistant", next_field["question"])
                    except Exception:
                        logger.exception("Failed to store assistant follow-up question (non-fatal)")
                    return QueryResponse(answer=next_field["question"], retrieved=[], context=None)

                completion_msg = "Thank you! Your details have been saved."
                try:
                    support_chat.store_message(session_id, "assistant", completion_msg)
                except Exception:
                    logger.exception("Failed to store onboarding completion message (non-fatal)")
                return QueryResponse(answer=completion_msg, retrieved=[], context=None)

            else:
                # Ask the first onboarding question
                try:
                    support_chat.store_message(session_id, "assistant", expected_question)
                except Exception:
                    logger.exception("Failed to store assistant onboarding question (non-fatal)")

                return QueryResponse(answer=expected_question, retrieved=[], context=None)

    # Build LLM prefix with profile if available
    if session_id:
        profile = get_all_user_meta(session_id)
        llm_prefix = support_chat.build_prompt_prefix(
            requester=requester,
            history=session_history,
            category=req.category,
        )

        if profile:
            prefix_extra_lines = ["User Profile:"]
            for k, v in profile.items():
                prefix_extra_lines.append(f"- {k}: {v}")
            prefix_extra = "\n".join(prefix_extra_lines) + "\n\n"
            llm_prefix = prefix_extra + llm_prefix

    # Execute RAG query
    try:
        if model_provider == "local":
            res = await rag_service.query_local_rag(
                query_text=req.question,
                n_results=req.top_k,
                requester=requester,
                llm_prompt_prefix=llm_prefix,
                use_llm=req.use_llm,
                max_tokens=req.max_tokens,
                session_id=session_id,
            )
        elif model_provider == "google":
            res = await query_google_rag(
                query_text=req.question,
                n_results=req.top_k,
                requester=requester,
                llm_prompt_prefix=llm_prefix,
                use_llm=req.use_llm,
            )
        else:
            raise HTTPException(status_code=400, detail=f"Invalid model provider: {model_provider}")
    except Exception as e:
        logger.exception("RAG query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    # Process results
    docs = []
    retrieved_docs = res.get("documents") or []
    metadatas = res.get("metadatas") or []
    ids = res.get("ids") or []
    distances = res.get("distances") or []

    if retrieved_docs and isinstance(retrieved_docs[0], list):
        retrieved_docs = retrieved_docs[0]
    if metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if distances and isinstance(distances[0], list):
        distances = distances[0]

    for i, doc_text in enumerate(retrieved_docs):
        meta = metadatas[i] if i < len(metadatas) else None
        id_ = ids[i] if i < len(ids) else f"doc_{i}"
        dist = distances[i] if i < len(distances) else None
        docs.append(RetrievedDoc(id=str(id_), text=doc_text, metadata=meta, distance=dist))

    answer = res.get("answer")
    if not answer:
        if docs:
            answer = "I found some relevant documents. Review the 'retrieved' items for details."
        else:
            answer = "No relevant documents found in the knowledge base."

    # Store conversation in session
    if session_id:
        try:
            support_chat.store_message(session_id, "user", req.question)
            support_chat.store_message(session_id, "assistant", answer)
        except Exception as exc:
            logger.exception("Failed to store session messages: %s", exc)

    return QueryResponse(answer=answer, retrieved=docs, context=res.get("context"))


@router.post("/add", response_model=AddResponse, dependencies=[Depends(require_roles(["SuperAdmin", "HR", "Manager", "Employee"]))])
async def add_document_json(
    req: AddDocRequest,
    requester: Dict[str, Any] = Depends(get_current_user),
    rag_service=Depends(get_rag_service)
):
    metadata = req.metadata or {}
    metadata.setdefault("department", metadata.get("department", "General"))
    metadata.setdefault("sensitivity", metadata.get("sensitivity", "public_internal"))
    metadata["ingested_by"] = requester.get("user_id")
    if "ingested_at" in metadata and metadata["ingested_at"] is None:
        del metadata["ingested_at"]

    validate_metadata(metadata)

    try:
        ids = await rag_service.add_document_to_rag_local(source_name=req.source_name, text=req.text, metadata=metadata)
    except Exception as e:
        logger.exception("Failed to add document: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    msg = f"Added {len(ids)} chunks for {req.source_name}"
    return AddResponse(message=msg, chunk_count=len(ids))


@router.post("/add-file", response_model=AddResponse, dependencies=[Depends(require_roles(["SuperAdmin", "HR", "Manager", "Employee"]))])
async def add_document_file(
    file: UploadFile = File(...),
    requester: Dict[str, Any] = Depends(get_current_user),
    department: Optional[str] = "General",
    sensitivity: Optional[str] = "public_internal",
    rag_service=Depends(get_rag_service)
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 5 MB).")

    try:
        text_content = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.exception("Failed to decode uploaded file: %s", e)
        raise HTTPException(status_code=400, detail="Failed to decode file; ensure it's a text file (UTF-8).")

    # Determine format from extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext in ['.md', '.markdown']:
        fmt = RawFormat.MARKDOWN
    elif ext in ['.html', '.htm']:
        fmt = RawFormat.HTML
    elif ext in ['.json']:
        fmt = RawFormat.JSON
    else:
        fmt = RawFormat.PLAIN

    try:
        # Parse text using doc_parser
        text = parse_text(text_content, format=fmt)
    except Exception as e:
        logger.warning(f"Failed to parse file {file.filename} as {fmt}, falling back to plain text. Error: {e}")
        text = text_content

    metadata = {"department": department, "sensitivity": sensitivity, "ingested_by": requester.get("user_id")}
    validate_metadata(metadata)

    try:
        ids = await rag_service.add_document_to_rag_local(source_name=file.filename, text=text, metadata=metadata)
    except Exception as e:
        logger.exception("Failed to add file: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    msg = f"Successfully ingested '{file.filename}'. {len(ids)} chunks created."
    return AddResponse(message=msg, chunk_count=len(ids))


@router.post("/seed", response_model=AddResponse, dependencies=[Depends(require_roles(["SuperAdmin"]))])
async def seed_defaults(
    requester: Dict[str, Any] = Depends(get_current_user),
    reseed: bool = False,
    rag_service=Depends(get_rag_service)
):
    try:
        ids = await rag_service.seed_from_file(force_reseed=reseed)
        if ids:
            return AddResponse(message=f"Seeded default docs from companyData. Chunks added: {len(ids)}.", chunk_count=len(ids))
        else:
            msg = "Seed operation skipped: collection not empty (use ?reseed=true to force) or no files found."
            return AddResponse(message=msg, chunk_count=0)

    except Exception as e:
        logger.exception("Seeding failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear", response_model=AddResponse, dependencies=[Depends(require_roles(["SuperAdmin"]))])
def clear_store(
    requester: Dict[str, Any] = Depends(get_current_user),
    rag_service=Depends(get_rag_service)
):
    # Remove the old role check since it's now handled by require_roles
    try:
        rag_service.clear_collection()
        return AddResponse(message="Collection cleared.", chunk_count=0)
    except Exception as e:
        logger.exception("Failed to clear collection: %s", e)
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/sentiment", dependencies=[Depends(require_roles(["SuperAdmin"]))])
def api_sentiment(req: SentimentRequest):
    classifier = get_global_sentiment()
    res = classifier.predict_single(req.text)
    return {"ok": True, "result": res}

@router.get("/sentiment/stats", dependencies=[Depends(require_roles(["SuperAdmin"]))])
def sentiment_stats_api():
    return get_sentiment_stats()
