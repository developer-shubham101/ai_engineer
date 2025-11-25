# app/api_routes_rag.py

from typing import List, Optional, Dict, Any
import logging
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Header
from pydantic import BaseModel, Field
from app.services.sentiment_classifier import get_global_sentiment
from app.services.support_chat import get_sentiment_stats

from app.services.rag_local_service import (
    add_document_to_rag_local,
    query_local_rag,
    seed_from_file,
    clear_collection,
)
from app.services.google_models import query_google_rag
from app.services.auth import get_user_from_api_key
from app.services import support_chat

from app.services.support_chat import (
    get_next_missing_profile_key,
    set_profile_value,
    get_full_profile,
)

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
# Simple auth dependency
# ---------------------------

def get_requester(x_api_key: Optional[str] = Header(None)) -> Dict[str, Any]:
    """
    Learning-mode auth:
    - Use an API key header 'X-API-Key' to simulate identity/role.
    - If missing/unknown, return a Guest user dict.
    """
    user = get_user_from_api_key(x_api_key) if x_api_key else None
    if not user:
        # Guest role (limited)
        return {"user_id": None, "role": "Guest", "department": None}
    return user


# ---------------------------
# Routes
# ---------------------------

@router.post("/{model_provider}/query", response_model=QueryResponse)
async def query(
    model_provider: str,
    req: QueryRequest,
    requester: Dict[str, Any] = Depends(get_requester),
    x_session_id: Optional[str] = Header(None),
):
    """
    Query the RAG. Supports session-aware onboarding and personalization.
    This endpoint can be used with different model providers (e.g., 'local', 'google').
    """
    logger.info("Query request: provider=%s, role=%s user=%s question=%s", model_provider, requester.get("role"), requester.get("user_id"), req.question)

    llm_prefix = None
    session_history = []

    if x_session_id:
        if not support_chat.session_exists(x_session_id):
            raise HTTPException(status_code=404, detail="Session not found. Start a new session first.")
        try:
            support_chat.touch_session(
                session_id=x_session_id,
                role=requester.get("role"),
                department=requester.get("department"),
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))

        session_history = support_chat.fetch_recent_messages(
            session_id=x_session_id,
            limit=support_chat.MAX_HISTORY_TURNS,
        )

    if x_session_id:
        profile = get_full_profile(x_session_id)
        next_field = get_next_missing_profile_key(x_session_id)

        if next_field:
            last_assistant_msg = None
            if session_history:
                last_msg = session_history[-1]
                if last_msg.get("speaker", "").lower() == "assistant":
                    last_assistant_msg = last_msg.get("content", "")

            expected_question = next_field["question"]
            key_to_save = next_field["key"]

            if last_assistant_msg and last_assistant_msg.strip() == expected_question.strip():
                user_reply = req.question.strip()

                try:
                    support_chat.store_message(x_session_id, "user", req.question)
                except Exception:
                    logger.exception("Failed to store user onboarding reply (non-fatal)")

                try:
                    set_profile_value(x_session_id, key_to_save, user_reply)
                except Exception as exc:
                    logger.exception("Failed to save onboarding value: %s", exc)
                    raise HTTPException(status_code=500, detail="Failed to save onboarding data.")

                next_field = get_next_missing_profile_key(x_session_id)
                if next_field:
                    try:
                        support_chat.store_message(x_session_id, "assistant", next_field["question"])
                    except Exception:
                        logger.exception("Failed to store assistant follow-up question (non-fatal)")
                    return QueryResponse(answer=next_field["question"], retrieved=[], context=None)

                completion_msg = "Thank you! Your details have been saved."
                try:
                    support_chat.store_message(x_session_id, "assistant", completion_msg)
                except Exception:
                    logger.exception("Failed to store onboarding completion message (non-fatal)")
                return QueryResponse(answer=completion_msg, retrieved=[], context=None)

            else:
                try:
                    support_chat.store_message(x_session_id, "assistant", expected_question)
                except Exception:
                    logger.exception("Failed to store assistant onboarding question (non-fatal)")

                return QueryResponse(answer=expected_question, retrieved=[], context=None)

    if x_session_id:
        profile = get_full_profile(x_session_id)
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

    try:
        if model_provider == "local":
            res = await query_local_rag(
                query_text=req.question,
                n_results=req.top_k,
                requester=requester,
                llm_prompt_prefix=llm_prefix,
                use_llm=req.use_llm,
                max_tokens=req.max_tokens,
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

    if x_session_id:
        try:
            support_chat.store_message(x_session_id, "user", req.question)
            support_chat.store_message(x_session_id, "assistant", answer)
        except Exception as exc:
            logger.exception("Failed to store session messages: %s", exc)

    return QueryResponse(answer=answer, retrieved=docs, context=res.get("context"))




@router.post("/add", response_model=AddResponse)
async def add_document_json(req: AddDocRequest, requester: Dict[str, Any] = Depends(get_requester)):
    metadata = req.metadata or {}
    metadata.setdefault("department", metadata.get("department", "General"))
    metadata.setdefault("sensitivity", metadata.get("sensitivity", "public_internal"))
    metadata["ingested_by"] = requester.get("user_id")
    if "ingested_at" in metadata and metadata["ingested_at"] is None:
        del metadata["ingested_at"]

    validate_metadata(metadata)

    try:
        ids = await add_document_to_rag_local(source_name=req.source_name, text=req.text, metadata=metadata)
    except Exception as e:
        logger.exception("Failed to add document: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    msg = f"Added {len(ids)} chunks for {req.source_name}"
    return AddResponse(message=msg, chunk_count=len(ids))


@router.post("/add-file", response_model=AddResponse)
async def add_document_file(
    file: UploadFile = File(...),
    requester: Dict[str, Any] = Depends(get_requester),
    department: Optional[str] = "General",
    sensitivity: Optional[str] = "public_internal",
):
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 5 MB).")

    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.exception("Failed to decode uploaded file: %s", e)
        raise HTTPException(status_code=400, detail="Failed to decode file; ensure it's a text file (UTF-8).")

    metadata = {"department": department, "sensitivity": sensitivity, "ingested_by": requester.get("user_id")}
    validate_metadata(metadata)

    try:
        ids = await add_document_to_rag_local(source_name=file.filename, text=text, metadata=metadata)
    except Exception as e:
        logger.exception("Failed to add file: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    msg = f"Successfully ingested '{file.filename}'. {len(ids)} chunks created."
    return AddResponse(message=msg, chunk_count=len(ids))


@router.post("/seed", response_model=AddResponse)
async def seed_defaults(
    requester: Dict[str, Any] = Depends(get_requester),
    reseed: bool = False
):
    try:
        ids = await seed_from_file(force_reseed=reseed)
        if ids:
            return AddResponse(message=f"Seeded default docs from companyData. Chunks added: {len(ids)}.", chunk_count=len(ids))
        else:
            msg = "Seed operation skipped: collection not empty (use ?reseed=true to force) or no files found."
            return AddResponse(message=msg, chunk_count=0)

    except Exception as e:
        logger.exception("Seeding failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear", response_model=AddResponse)
def clear_store(requester: Dict[str, Any] = Depends(get_requester)):
    role = requester.get("role")
    if role not in ("Executive", "Legal"):
        raise HTTPException(status_code=403, detail="Clearing the collection is restricted to Executive/Legal in this demo.")

    try:
        clear_collection()
        return AddResponse(message="Collection cleared.", chunk_count=0)
    except Exception as e:
        logger.exception("Failed to clear collection: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/session/start", response_model=SupportSessionStartResponse)
async def start_support_session(requester: Dict[str, Any] = Depends(get_requester)):
    session_id = f"sess_{uuid.uuid4().hex}"

    try:
        support_chat.create_session(
            session_id=session_id,
            role=requester.get("role"),
            department=requester.get("department")
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return SupportSessionStartResponse(
        session_id=session_id,
        message="Session started"
    )


@router.post("/session/end", response_model=SupportSessionEndResponse)
async def end_support_session(req: SupportSessionEndRequest, requester: Dict[str, Any] = Depends(get_requester)):
    if not support_chat.session_exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        support_chat.end_session(req.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return SupportSessionEndResponse(session_id=req.session_id, message="Support session ended.")


@router.post("/sentiment")
def api_sentiment(req: SentimentRequest):
    classifier = get_global_sentiment()
    res = classifier.predict_single(req.text)
    return {"ok": True, "result": res}

@router.get("/sentiment/stats")
def sentiment_stats_api():
    return get_sentiment_stats()
