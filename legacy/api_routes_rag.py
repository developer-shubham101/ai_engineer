# app/api_routes_rag.py

from typing import List, Optional, Dict, Any
import logging
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel, Field
from app.services.legacy.sentiment_classifier import get_global_sentiment
from app.services.legacy.support_chat import get_sentiment_stats

# Import dependency providers
from app.dependencies import get_rag_service, get_current_user, get_current_user_optional, require_roles
from app.services.legacy.user_service import get_all_user_meta

# from app.services.google_models import query_google_rag
# from app.services.gpt_rag_service import query_gpt_rag
# from app.services.hf_rag_service import query_hf_rag
from app.services.legacy import support_chat

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
    temperature: float = 0.1  # Default temperature for all providers
    category: Optional[str] = None
    debug: bool = False
    local_llm_model: Optional[str] = None  # Model selection for local provider


class QueryResponse(BaseModel):
    answer: Optional[str] = None
    retrieved: List[RetrievedDoc] = Field(default_factory=list)
    context: Optional[str] = None
    final_prompt: Optional[str] = None

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
    "super_confidential",
    "personal",
}

ALLOWED_DEPARTMENTS = {
    "General",
    "HR",
    "Finance",
    "Engineering",
    "IT",
    "Legal",
    "Executive",
    "Admin",
}

# Flexible Role System: Level-based + Specific Role Overrides
ALLOWED_ROLES = {
    "SuperAdmin",     # Level 4
    "Manager",       # Level 3
    "HR",            # Level 2  
    "Employee",      # Level 1
    "PublicUser",    # Level 0
    "Guest",         # Level 0
}

# Role hierarchy levels - higher numbers = more access
ROLE_LEVELS = {
    "SuperAdmin": 4,    # Level 4 - Full access
    "Manager": 3,      # Level 3 - Management access
    "HR": 2,           # Level 2 - HR access
    "Employee": 1,     # Level 1 - Employee access
    "PublicUser": 0,   # Level 0 - Public only
    "Guest": 0,        # Level 0 - Public only
}

# Sensitivity level requirements
SENSITIVITY_LEVELS = {
    "public_internal": 0,        # Anyone
    "department_confidential": 1, # Employee+
    "role_confidential": 2,      # HR+
    "highly_confidential": 3,    # Manager+
    "super_confidential": 4,     # SuperAdmin only
    "personal": 1,               # Employee+ (with ownership check)
}


def validate_metadata(meta: Optional[Dict[str, Any]], requester: Optional[Dict[str, Any]] = None):
    """
    Validate document metadata with flexible RBAC support.
    """
    if not meta:
        return
    
    # Validate sensitivity level
    sens = meta.get("sensitivity")
    if sens and sens not in ALLOWED_SENSITIVITY:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid sensitivity '{sens}'. Allowed: {list(ALLOWED_SENSITIVITY)}"
        )
    
    # Validate department
    dept = meta.get("department")
    if dept and dept not in ALLOWED_DEPARTMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid department '{dept}'. Allowed: {list(ALLOWED_DEPARTMENTS)}"
        )
    
    # Validate allowed_roles if present
    allowed_roles = meta.get("allowed_roles")
    if allowed_roles:
        if not isinstance(allowed_roles, list):
            raise HTTPException(
                status_code=400,
                detail="allowed_roles must be a list of role names"
            )
        invalid_roles = [r for r in allowed_roles if r not in ALLOWED_ROLES]
        if invalid_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid roles in allowed_roles: {invalid_roles}. Allowed: {list(ALLOWED_ROLES)}"
            )
    
    # Level-based sensitivity validation
    if requester and sens:
        user_role = requester.get("role")
        user_level = ROLE_LEVELS.get(user_role, 0)
        required_level = SENSITIVITY_LEVELS.get(sens, 0)
        
        if user_level < required_level:
            raise HTTPException(
                status_code=403,
                detail=f"Your role '{user_role}' (level {user_level}) cannot create documents with sensitivity '{sens}' (requires level {required_level}+)"
            )
    
    # Validate personal documents have owner_id
    if sens == "personal" and not meta.get("owner_id"):
        raise HTTPException(
            status_code=400,
            detail="Personal documents must have an 'owner_id' field"
        )



@router.post("/{model_provider}/query", response_model=QueryResponse)
async def query_rag(
    model_provider: str,
    req: QueryRequest,
    requester: Optional[Dict[str, Any]] = Depends(get_current_user_optional),
    rag_service=Depends(get_rag_service),
):
    """
    RAG query endpoint with three flows:
    1. Authenticated Company User (non-Guest role): Fetch profile from user_meta, no onboarding
    2. Authenticated Guest User (Guest role): Ask onboarding questions, use session_id from token
    3. Unauthenticated User: Basic RAG without onboarding or session tracking
    """
    
    # Add debug flag support
    debug_mode = getattr(req, "debug", False)
    
    # Determine user context
    user_id = requester.get("user_id") if requester else None
    user_role = requester.get("role") if requester else None
    session_id = requester.get("session_id") if requester else None
    
    # CASE 1: AUTHENTICATED COMPANY USER (non-Guest role)
    if user_id and user_role and user_role != "Guest":
        # Fetch profile from user_meta using user_id (persistent profile)
        profile = get_all_user_meta(user_id)
        
        # Ensure session exists in support_sessions.db for history
        if session_id and not support_chat.session_exists(session_id):
            try:
                support_chat.create_session(
                    session_id=session_id,
                    role=user_role,
                    department=requester.get("department")
                )
            except Exception:
                pass # Session might already exist
        
        # Touch session to update timestamp
        if session_id:
            try:
                support_chat.touch_session(
                    session_id=session_id,
                    role=user_role,
                    department=requester.get("department"),
                )
            except ValueError:
                pass

        # Fetch conversation history
        session_history = support_chat.fetch_recent_messages(
            session_id=session_id,
            limit=support_chat.MAX_HISTORY_TURNS,
        ) if session_id else []
        
        # NO ONBOARDING for company users

    # CASE 2: AUTHENTICATED GUEST USER (Guest role with token)
    elif user_id and user_role == "Guest":
        session_history = []
        profile = {}
        
        # Ensure guest session exists using session_id from token
        if session_id:
            if not support_chat.session_exists(session_id):
                try:
                    support_chat.create_session(
                        session_id=session_id,
                        role="Guest",
                        department="General"
                    )
                except Exception:
                    pass

            # Fetch conversation history
            session_history = support_chat.fetch_recent_messages(
                session_id=session_id,
                limit=support_chat.MAX_HISTORY_TURNS,
            )
            
            # Handle Onboarding for Guest users
            next_field = support_chat.get_next_missing_profile_key(session_id)
            
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

                    # Save to session_profiles
                    try:
                        support_chat.set_profile_value(session_id, key_to_save, user_reply)
                    except Exception as exc:
                        logger.exception("Failed to save onboarding value: %s", exc)
                        raise HTTPException(status_code=500, detail="Failed to save onboarding data.")

                    # Check for next missing field
                    next_field = support_chat.get_next_missing_profile_key(session_id)
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

            # Get full profile for context
            profile = support_chat.get_full_profile(session_id)

    # CASE 3: UNAUTHENTICATED USER (no token)
    else:
        # Basic RAG without onboarding or session tracking
        session_id = None
        profile = {}
        session_history = []


    # Use optimized prompt building - pass None to let base service handle it
    llm_prefix = None  # Let the base RAG service handle prompt optimization

    # Execute RAG query (Business Requirement Step 2 & 3: Query + Role Check)
    try:
        if model_provider == "local":
            res = await rag_service.query_local_rag(
                query_text=req.question,
                n_results=req.top_k,
                requester=requester,  # Step 3: Role evaluation happens in base service
                llm_prompt_prefix=llm_prefix,
                use_llm=req.use_llm,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                session_id=session_id,
                model_key=req.local_llm_model,
            )
        elif model_provider == "google":
            res = await query_google_rag(
                query_text=req.question,
                n_results=req.top_k,
                requester=requester,
                llm_prompt_prefix=llm_prefix,
                use_llm=req.use_llm,
                temperature=req.temperature,
                session_id=session_id,
            )
        elif model_provider == "gpt":
            res = await query_gpt_rag(
                query_text=req.question,
                n_results=req.top_k,
                requester=requester,
                llm_prompt_prefix=llm_prefix,
                use_llm=req.use_llm,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                session_id=session_id,
            )
        elif model_provider == "huggingface" or model_provider == "hf":
            res = await query_hf_rag(
                query_text=req.question,
                n_results=req.top_k,
                requester=requester,
                llm_prompt_prefix=llm_prefix,
                use_llm=req.use_llm,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                session_id=session_id,
            )
        else:
            supported_providers = ["local", "google", "gpt", "huggingface", "hf"]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model provider: {model_provider}. Supported providers: {supported_providers}"
            )
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

    return QueryResponse(
        answer=answer, 
        retrieved=docs, 
        context=res.get("context"),
        final_prompt=res.get("final_prompt")
    )


@router.post("/documents/add", response_model=AddResponse, dependencies=[Depends(require_roles(["SuperAdmin", "Manager", "HR", "Employee"]))])
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

    try:
        validate_metadata(metadata, requester)  # Pass requester for role-based validation
    except HTTPException as e:
        # Log validation failures for audit
        from app.logging_config import log_security_event
        log_security_event(
            logger, "METADATA_VALIDATION_FAILED", requester.get("user_id"),
            role=requester.get("role"), department=requester.get("department"),
            error=e.detail, source_name=req.source_name
        )
        raise

    try:
        result = await rag_service.add_document_to_rag_local(
            source_name=req.source_name, 
            text=req.text, 
            metadata=metadata,
            created_by=requester.get("user_id")
        )
        
        # Log successful creation
        from app.logging_config import log_user_action
        log_user_action(
            logger, "DOCUMENT_CREATED", requester.get("user_id"),
            role=requester.get("role"), department=requester.get("department"),
            document_id=result["document_id"], sensitivity=metadata.get("sensitivity"),
            chunk_count=result["chunk_count"], version=result["version"]
        )
    except Exception as e:
        logger.exception("Failed to add document: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    msg = f"Added document '{req.source_name}' (v{result['version']}, {result['chunk_count']} chunks, document_id={result['document_id']})"
    return AddResponse(message=msg, chunk_count=result['chunk_count'])


@router.post("/documents/add-file", response_model=AddResponse, dependencies=[Depends(require_roles(["SuperAdmin", "Manager", "HR", "Employee"]))])
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
    
    try:
        validate_metadata(metadata, requester)  # Pass requester for role-based validation
    except HTTPException as e:
        # Log validation failures for audit
        from app.logging_config import log_security_event
        log_security_event(
            logger, "FILE_UPLOAD_VALIDATION_FAILED", requester.get("user_id"),
            role=requester.get("role"), department=requester.get("department"),
            filename=file.filename, error=e.detail, file_size=len(raw)
        )
        raise

    try:
        result = await rag_service.add_document_to_rag_local(
            source_name=file.filename, 
            text=text, 
            metadata=metadata,
            created_by=requester.get("user_id")
        )
        
        # Log successful file upload
        from app.logging_config import log_user_action
        log_user_action(
            logger, "FILE_UPLOADED", requester.get("user_id"),
            role=requester.get("role"), department=requester.get("department"),
            filename=file.filename, document_id=result["document_id"], 
            sensitivity=metadata.get("sensitivity"), chunk_count=result["chunk_count"],
            file_size=len(raw), format=fmt.value if hasattr(fmt, 'value') else str(fmt)
        )
    except Exception as e:
        logger.exception("Failed to add file: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    msg = f"Successfully ingested '{file.filename}' (v{result['version']}, {result['chunk_count']} chunks, document_id={result['document_id']})"
    return AddResponse(message=msg, chunk_count=result['chunk_count'])


@router.post("/documents/seed", response_model=AddResponse, dependencies=[Depends(require_roles(["SuperAdmin"]))])
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


@router.post("/documents/clear", response_model=AddResponse, dependencies=[Depends(require_roles(["SuperAdmin"]))])
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


# ---------------------------
# Document Versioning Endpoints
# ---------------------------

class UpdateDocumentRequest(BaseModel):
    document_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None
    version_notes: Optional[str] = None
    status: str = "published"


class UpdateDocumentResponse(BaseModel):
    message: str
    document_id: str
    version: str
    chunk_count: int
    status: str


class DocumentListResponse(BaseModel):
    documents: List[Dict[str, Any]]
    count: int


class VersionHistoryResponse(BaseModel):
    document_id: str
    versions: List[Dict[str, Any]]


class DocumentVersionResponse(BaseModel):
    document_id: str
    version: str
    source_name: str
    chunks: List[str]
    created_at: str
    created_by: Optional[str]
    status: str
    version_notes: Optional[str]


class CompareVersionsResponse(BaseModel):
    document_id: str
    version1: str
    version2: str
    diff: str
    summary: Dict[str, Any]


@router.post("/documents/update", response_model=UpdateDocumentResponse, 
             dependencies=[Depends(require_roles(["SuperAdmin", "Manager", "HR", "Employee"]))])
async def update_document(
    req: UpdateDocumentRequest,
    requester: Dict[str, Any] = Depends(get_current_user),
    rag_service=Depends(get_rag_service)
):
    """
    Update a document by creating a new version (non-destructive).
    
    Validates:
    - User has permission to update documents with the new sensitivity level
    - User's department matches document's department (or is SuperAdmin/HR)
    - Metadata is valid
    """
    try:
        # Get current document to check ownership/department
        from app.services.legacy import version_tracking
        latest_version = version_tracking.get_latest_version(req.document_id)
        if not latest_version:
            raise HTTPException(status_code=404, detail=f"Document {req.document_id} not found")
        
        current_metadata = latest_version.get("metadata", {})
        current_dept = current_metadata.get("department", "General")
        user_role = requester.get("role")
        user_dept = requester.get("department")
        
        # Department ownership check (unless high-level roles)
        user_level = ROLE_LEVELS.get(user_role, 0)
        if user_level < 2:  # Below HR level
            if current_dept != user_dept:
                from app.logging_config import log_security_event
                log_security_event(
                    logger, "RBAC_UPDATE_DENIED", requester.get("user_id"),
                    role=user_role, user_dept=user_dept, document_dept=current_dept,
                    document_id=req.document_id
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Your role '{user_role}' cannot update documents from department '{current_dept}'. Your department is '{user_dept}'."
                )
        
        # Validate new metadata if provided
        if req.metadata:
            # Merge with existing metadata
            updated_metadata = {**current_metadata, **req.metadata}
            validate_metadata(updated_metadata, requester)
            
            # Log metadata changes
            if req.metadata.get("sensitivity") and req.metadata["sensitivity"] != current_metadata.get("sensitivity"):
                from app.logging_config import log_user_action
                log_user_action(
                    logger, "METADATA_SENSITIVITY_CHANGED", requester.get("user_id"),
                    document_id=req.document_id, 
                    old_sensitivity=current_metadata.get("sensitivity"),
                    new_sensitivity=req.metadata["sensitivity"],
                    role=requester.get("role")
                )
        else:
            # Use existing metadata
            updated_metadata = current_metadata
        
        # Perform update
        result = await rag_service.update_document_version(
            document_id=req.document_id,
            text=req.text,
            metadata=updated_metadata,
            version_notes=req.version_notes,
            requester_id=requester.get("user_id"),
            status=req.status
        )
        
        from app.logging_config import log_user_action
        log_user_action(
            logger, "DOCUMENT_UPDATED", requester.get("user_id"),
            document_id=req.document_id, new_version=result["version"],
            parent_version=parent_version, chunk_count=result["chunk_count"],
            status=req.status, has_notes=bool(req.version_notes)
        )
        
        return UpdateDocumentResponse(
            message=f"Created version {result['version']} for document {req.document_id}",
            document_id=result["document_id"],
            version=result["version"],
            chunk_count=result["chunk_count"],
            status=req.status
        )
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Failed to update document: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/list", response_model=DocumentListResponse,
            dependencies=[Depends(require_roles(["SuperAdmin", "Manager", "HR", "Employee"]))])
async def list_all_documents(
    department: Optional[str] = None,
    status: Optional[str] = None,
    latest_only: bool = True,
    rag_service=Depends(get_rag_service)
):
    """
    List all documents with optional filtering.
    """
    try:
        documents = await rag_service.list_documents(
            department=department,
            status=status,
            latest_only=latest_only
        )
        return DocumentListResponse(documents=documents, count=len(documents))
    except Exception as e:
        logger.exception("Failed to list documents: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/versions", response_model=VersionHistoryResponse,
            dependencies=[Depends(require_roles(["SuperAdmin", "Manager", "HR", "Employee"]))])
async def get_version_history(
    document_id: str,
    rag_service=Depends(get_rag_service)
):
    """
    Get version history for a document.
    """
    try:
        from app.services.legacy import version_tracking
        versions = version_tracking.get_version_history(document_id)
        return VersionHistoryResponse(document_id=document_id, versions=versions)
    except Exception as e:
        logger.exception("Failed to get version history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/versions/{version}", response_model=DocumentVersionResponse,
            dependencies=[Depends(require_roles(["SuperAdmin", "Manager", "HR", "Employee"]))])
async def get_specific_version(
    document_id: str,
    version: str,
    rag_service=Depends(get_rag_service)
):
    """
    Get a specific version of a document.
    """
    try:
        result = await rag_service.get_document_version(document_id, version)
        if not result:
            raise HTTPException(status_code=404, detail=f"Version {version} of document {document_id} not found")
        
        return DocumentVersionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get document version: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/compare", response_model=CompareVersionsResponse,
            dependencies=[Depends(require_roles(["SuperAdmin", "Manager", "HR", "Employee"]))])
async def compare_versions(
    document_id: str,
    version1: str,
    version2: str,
    rag_service=Depends(get_rag_service)
):
    """
    Compare two versions of a document.
    """
    try:
        result = await rag_service.compare_document_versions(document_id, version1, version2)
        if not result:
            raise HTTPException(status_code=404, detail="One or both versions not found")
        
        return CompareVersionsResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to compare versions: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/{document_id}/archive", response_model=AddResponse,
             dependencies=[Depends(require_roles(["SuperAdmin", "Manager", "HR"]))]) 
async def archive_version(
    document_id: str,
    version: str,
    rag_service=Depends(get_rag_service)
):
    """
    Archive a specific version of a document.
    """
    try:
        success = await rag_service.archive_document_version(document_id, version)
        if not success:
            raise HTTPException(status_code=404, detail=f"Version {version} of document {document_id} not found")
        
        return AddResponse(message=f"Archived version {version} of document {document_id}", chunk_count=0)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to archive version: %s", e)
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/sentiment", dependencies=[Depends(require_roles(["SuperAdmin"]))])
def api_sentiment(req: SentimentRequest):
    classifier = get_global_sentiment()
    res = classifier.predict_single(req.text)
    return {"ok": True, "result": res}

@router.get("/sentiment/stats", dependencies=[Depends(require_roles(["SuperAdmin"]))])
def sentiment_stats_api():
    return get_sentiment_stats()


# ---------------------------
# Embedding Model Status Endpoint
# ---------------------------

@router.get("/embedding/status", dependencies=[Depends(require_roles(["SuperAdmin", "Manager"]))])
def embedding_model_status():
    """
    Get current embedding model status and configuration.
    Endpoint for monitoring embedding model performance.
    """
    from app.services.utility import get_embedding_model_info
    
    try:
        info = get_embedding_model_info()
        from app.logging_config import log_user_action
        log_user_action(
            logger, "EMBEDDING_STATUS_CHECK", "system",
            model_key=info.get("model_key"), model_loaded=info.get("model_loaded"),
            dimensions=info.get("actual_dimensions"), model_size=info.get("model_size", "unknown")
        )
        return {"ok": True, "embedding_model": info}
    except Exception as e:
        from app.logging_config import log_security_event
        log_security_event(logger, "EMBEDDING_STATUS_ERROR", error=str(e))
        return {"ok": False, "error": str(e)}
