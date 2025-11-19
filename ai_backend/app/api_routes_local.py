# app/api_routes_local.py

from typing import List, Optional, Dict, Any
import logging
import uuid
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Header
from pydantic import BaseModel, Field

from app.services.rag_local_service import (
    add_document_to_rag_local,
    query_local_rag,
    seed_from_file,
    clear_collection,
)
# simple auth map service (create app/services/auth.py if not present)
from app.services.auth import get_user_from_api_key  # expects a small mapping
from app.services import support_chat


from app.services.support_chat import (
    get_next_missing_profile_key,
    set_profile_value,
    get_full_profile,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/local", tags=["Local RAG"])
support_chat.init_support_chat_db(reset_on_start=True)

# ---------------------------
# Models
# ---------------------------

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

@router.post("/query", response_model=QueryResponse)
async def query_local(
    req: QueryRequest,
    requester: Dict[str, Any] = Depends(get_requester),
    x_session_id: Optional[str] = Header(None),
):
    """
    Query the local RAG. Supports session-aware onboarding and personalization.

    Behavior summary:
    - If X-Session-Id is missing -> stateless RAG query (Scenario 3).
    - If X-Session-Id provided and onboarding incomplete -> ask onboarding questions sequentially (Scenario 1).
      * We detect whether the user's incoming message is an answer to the last assistant onboarding question
        by inspecting recent session history. If the last assistant message equals the onboarding question,
        we treat the current user text as the answer and save it.
    - If X-Session-Id provided and onboarding complete -> run RAG with personalized prefix (Scenario 2).
    """
    logger.info("Query request: role=%s user=%s question=%s", requester.get("role"), requester.get("user_id"), req.question)

    llm_prefix = None
    session_history = []

    # If session header provided, validate session and assemble history/prefix
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

        # fetch recent history to (a) build prompt prefix and (b) detect onboarding flow
        session_history = support_chat.fetch_recent_messages(
            session_id=x_session_id,
            limit=support_chat.MAX_HISTORY_TURNS,
        )

    # ------------------------
    # ONBOARDING / PROFILE LOGIC
    # ------------------------
    if x_session_id:
        # Load current short profile and find next missing onboarding field (from config)
        profile = get_full_profile(x_session_id)
        next_field = get_next_missing_profile_key(x_session_id)

        if next_field:
            # Determine whether the user's current request is an answer to the previous assistant onboarding question.
            last_assistant_msg = None
            if session_history:
                last_msg = session_history[-1]
                if last_msg.get("speaker", "").lower() == "assistant":
                    last_assistant_msg = last_msg.get("content", "")

            expected_question = next_field["question"]
            key_to_save = next_field["key"]

            # CASE A: If the assistant previously asked the expected onboarding question,
            # treat current req.question as the answer and save it.
            if last_assistant_msg and last_assistant_msg.strip() == expected_question.strip():
                user_reply = req.question.strip()

                # Save user's reply message to history
                try:
                    support_chat.store_message(x_session_id, "user", req.question)
                except Exception:
                    logger.exception("Failed to store user onboarding reply (non-fatal)")

                # Save profile key-value
                try:
                    set_profile_value(x_session_id, key_to_save, user_reply)
                except Exception as exc:
                    logger.exception("Failed to save onboarding value: %s", exc)
                    raise HTTPException(status_code=500, detail="Failed to save onboarding data.")

                # Re-evaluate next missing field after saving
                next_field = get_next_missing_profile_key(x_session_id)
                if next_field:
                    # store assistant's next question before returning
                    try:
                        support_chat.store_message(x_session_id, "assistant", next_field["question"])
                    except Exception:
                        logger.exception("Failed to store assistant follow-up question (non-fatal)")
                    return QueryResponse(answer=next_field["question"], retrieved=[], context=None)

                # Onboarding completed — store assistant completion message and return
                completion_msg = "Thank you! Your details have been saved."
                try:
                    support_chat.store_message(x_session_id, "assistant", completion_msg)
                except Exception:
                    logger.exception("Failed to store onboarding completion message (non-fatal)")
                return QueryResponse(answer=completion_msg, retrieved=[], context=None)

            # CASE B: The onboarding question has not been asked yet in this session — ask it now.
            else:
                # Persist the assistant question so the next user reply can be recognized as an answer
                try:
                    support_chat.store_message(x_session_id, "assistant", expected_question)
                except Exception:
                    logger.exception("Failed to store assistant onboarding question (non-fatal)")

                return QueryResponse(answer=expected_question, retrieved=[], context=None)

    # ------------------------
    # BUILD PERSONALIZED PREFIX (if session/profile exists)
    # ------------------------
    if x_session_id:
        profile = get_full_profile(x_session_id)
        llm_prefix = support_chat.build_prompt_prefix(
            requester=requester,
            history=session_history,
            category=req.category,
        )

        # Prepend profile details if available
        if profile:
            prefix_extra_lines = ["User Profile:"]
            for k, v in profile.items():
                prefix_extra_lines.append(f"- {k}: {v}")
            prefix_extra = "\n".join(prefix_extra_lines) + "\n\n"
            llm_prefix = prefix_extra + llm_prefix

    # ------------------------
    # CALL RAG SERVICE
    # ------------------------
    try:
        res = query_local_rag(
            query_text=req.question,
            n_results=req.top_k,
            requester=requester,       # pass role info for filtering in service
            llm_prompt_prefix=llm_prefix,
            use_llm=req.use_llm,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        logger.exception("RAG query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    # ------------------------
    # NORMALIZE RETRIEVED ITEMS
    # ------------------------
    docs = []
    retrieved_docs = res.get("documents") or []
    metadatas = res.get("metadatas") or []
    ids = res.get("ids") or []
    distances = res.get("distances") or []

    # handle nested lists from some Chroma responses
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
    # Friendly fallback if no LLM answer present
    if not answer:
        if docs:
            answer = "I found some relevant documents. Review the 'retrieved' items for details."
        else:
            answer = "No relevant documents found in the knowledge base."

    # ------------------------
    # STORE MESSAGES (if sessioned)
    # ------------------------
    if x_session_id:
        try:
            # store the user's question (if not already stored during onboarding branch)
            # Note: onboarding branch already stores user message when saving answers.
            # This will prevent duplicate storage for onboarding answers because store_message is idempotent in intent.
            support_chat.store_message(x_session_id, "user", req.question)
            support_chat.store_message(x_session_id, "assistant", answer)
        except Exception as exc:
            logger.exception("Failed to store session messages: %s", exc)

    return QueryResponse(answer=answer, retrieved=docs, context=res.get("context"))




@router.post("/add", response_model=AddResponse)
async def add_document_json(req: AddDocRequest, requester: Dict[str, Any] = Depends(get_requester)):
    """
    Add a document provided inline as JSON. Attach ingest metadata automatically.
    """
    metadata = req.metadata or {}
    # attach ingest metadata
    metadata.setdefault("department", metadata.get("department", "General"))
    metadata.setdefault("sensitivity", metadata.get("sensitivity", "public_internal"))
    metadata["ingested_by"] = requester.get("user_id")
    if "ingested_at" in metadata and metadata["ingested_at"] is None:  # optional if caller provided
        del metadata["ingested_at"]

    validate_metadata(metadata)

    try:
        ids = add_document_to_rag_local(source_name=req.source_name, text=req.text, metadata=metadata)
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
    """
    Upload a text file to ingest. For learning/demo we accept text files and limit size.
    Query string params 'department' and 'sensitivity' allow tagging at ingest time.
    """
    # Basic file size limit (5 MB) to keep local runs safe
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 5 MB).")

    # decode best-effort. For learning keep it simple: assume text files.
    try:
        text = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.exception("Failed to decode uploaded file: %s", e)
        raise HTTPException(status_code=400, detail="Failed to decode file; ensure it's a text file (UTF-8).")

    metadata = {"department": department, "sensitivity": sensitivity, "ingested_by": requester.get("user_id")}
    validate_metadata(metadata)

    try:
        ids = add_document_to_rag_local(source_name=file.filename, text=text, metadata=metadata)
    except Exception as e:
        logger.exception("Failed to add file: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    msg = f"Successfully ingested '{file.filename}'. {len(ids)} chunks created."
    return AddResponse(message=msg, chunk_count=len(ids))


@router.post("/seed", response_model=AddResponse)
def seed_defaults(requester: Dict[str, Any] = Depends(get_requester)):
    """
    Seed default file (if present). Uses rag_local_service.seed_from_file().
    This function will attempt to read the default path configured in the service.
    """
    try:
        ids = seed_from_file()
        return AddResponse(message="Seeded default docs (if any).", chunk_count=len(ids))
    except Exception as e:
        logger.exception("Seeding failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/clear", response_model=AddResponse)
def clear_store(requester: Dict[str, Any] = Depends(get_requester)):
    """
    Clear Chroma collection. For learning/demo only — in real deploy restrict to admins.
    """
    # simple guard: only Exec / Legal can clear the store in this learning example
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
    # >>> START OF SUGGESTED CODE ADDITION <<<
    # Auto-create session (no request body required)
    session_id = f"sess_{uuid.uuid4().hex}"

    # Store new session in SQLite DB
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
    # >>> END OF SUGGESTED CODE ADDITION <<<


@router.post("/session/end", response_model=SupportSessionEndResponse)
async def end_support_session(req: SupportSessionEndRequest, requester: Dict[str, Any] = Depends(get_requester)):
    if not support_chat.session_exists(req.session_id):
        raise HTTPException(status_code=404, detail="Session not found.")

    try:
        support_chat.end_session(req.session_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    return SupportSessionEndResponse(session_id=req.session_id, message="Support session ended.")
