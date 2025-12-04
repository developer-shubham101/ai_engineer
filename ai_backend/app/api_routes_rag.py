# app/api_routes_rag.py
"""
RAG API routes - refactored to use modular architecture.
"""
import logging
import os
from typing import List, Optional, Dict, Any

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel, Field

from app.dependencies import get_current_user, get_current_user_optional, require_roles
from app.modules.auth.interfaces import ISessionManager
from app.modules.config import MANAGER_PLUS_ROLES
from app.modules.core.document_manager import DocumentManager
from app.modules.integration import get_container
from app.modules.llm.rag_orchestrator import RAGOrchestrator
from app.modules.llm.interfaces import RAGRequest
from app.modules.config.constants import (
    VALID_SENSITIVITY_LEVELS, VALID_DEPARTMENTS, VALID_ROLES, ROLE_LEVELS, SENSITIVITY_LEVELS,
    DEFAULT_DEPARTMENT, DEFAULT_SENSITIVITY, DEFAULT_TOP_K, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE,
    MAX_FILE_SIZE_BYTES, HTTP_MESSAGES, HR_LEVEL_THRESHOLD, EMPLOYEE_PLUS_ROLES, SUPER_ADMIN_ROLES,
    HR_PLUS_ROLES, MARKDOWN_EXTENSIONS, HTML_EXTENSIONS, JSON_EXTENSIONS
)
from app.utils.doc_parser import parse_text, RawFormat

logger = logging.getLogger(__name__)
from app.modules.config.constants import RAG_PREFIX

router = APIRouter(prefix=RAG_PREFIX, tags=["RAG"])


# ---------------------------
# Models (keep original)
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
    top_k: int = DEFAULT_TOP_K
    use_llm: bool = False
    use_documents: bool = True  # Flag to control document retrieval
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    category: Optional[str] = None
    debug: bool = False
    local_llm_model: Optional[str] = None


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


from app.modules.config.constants import DEFAULT_STATUS

class UpdateDocumentRequest(BaseModel):
    document_id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None
    version_notes: Optional[str] = None
    status: str = DEFAULT_STATUS


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





def validate_metadata(meta: Optional[Dict[str, Any]], requester: Optional[Dict[str, Any]] = None) -> None:
    """Validate document metadata with RBAC support."""
    if not meta:
        return

    sens = meta.get("sensitivity")
    if sens and sens not in VALID_SENSITIVITY_LEVELS:
        raise HTTPException(status_code=400,
                            detail=f"Invalid sensitivity '{sens}'. Allowed: {list(VALID_SENSITIVITY_LEVELS)}")

    dept = meta.get("department")
    if dept and dept not in VALID_DEPARTMENTS:
        raise HTTPException(status_code=400,
                            detail=f"Invalid department '{dept}'. Allowed: {list(VALID_DEPARTMENTS)}")

    if requester and sens:
        user_role = requester.get("role")
        user_level = ROLE_LEVELS.get(user_role, 0)
        required_level = SENSITIVITY_LEVELS.get(sens, 0)

        if user_level < required_level:
            raise HTTPException(status_code=403,
                                detail=f"Your role '{user_role}' (level {user_level}) cannot create documents with sensitivity '{sens}' (requires level {required_level}+)")


def get_document_instance() -> DocumentManager:
    # Get modular services for user management
    container = get_container()
    container.initialize()

    document_manager: DocumentManager = container.get_document_manager()
    return document_manager


# ---------------------------
# Main Query Endpoints (use original working logic)
# ---------------------------
@router.post("/{model_provider}/query", response_model=QueryResponse)
async def query_rag(
        model_provider: str,
        req: QueryRequest,
        requester: Optional[Dict[str, Any]] = Depends(get_current_user_optional)
):
    """RAG query endpoint - refactored to use modular RAG Orchestrator."""

    container = get_container()
    container.initialize()
    rag_orchestrator: RAGOrchestrator = container.get_rag_orchestrator()

    # Create RAGRequest
    rag_request = RAGRequest(
        question=req.question,
        user=requester,
        session_id=requester.get("session_id") if requester else None,
        top_k=req.top_k,
        use_llm=req.use_llm,
        use_documents=req.use_documents,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        category=req.category,
        debug=req.debug,
        provider=model_provider,
        provider_specific={"model_name": req.local_llm_model} if req.local_llm_model else None
    )

    try:
        res = await rag_orchestrator.process_query(rag_request)

        # Process results
        docs = []
        for doc in res.retrieved_documents:
            docs.append(RetrievedDoc(id=doc.id, text=doc.text, metadata=doc.metadata, distance=doc.distance))

        return QueryResponse(
            answer=res.answer,
            retrieved=docs,
            context=res.context,
            final_prompt=res.final_prompt
        )
    except Exception as e:
        logger.exception("RAG query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# Document Management (use original working services)
# ---------------------------
@router.post("/documents/add", response_model=AddResponse,
             dependencies=[Depends(require_roles(EMPLOYEE_PLUS_ROLES))])
async def add_document_json(
        req: AddDocRequest,
        requester: Dict[str, Any] = Depends(get_current_user)
):
    """Add document via JSON - original working implementation."""
    metadata = req.metadata or {}
    metadata.setdefault("department", metadata.get("department", DEFAULT_DEPARTMENT))
    metadata.setdefault("sensitivity", metadata.get("sensitivity", DEFAULT_SENSITIVITY))
    metadata["ingested_by"] = requester.get("user_id")
    if "ingested_at" in metadata and metadata["ingested_at"] is None:
        del metadata["ingested_at"]

    try:
        validate_metadata(metadata, requester)
    except HTTPException as e:
        from app.logging_config import log_security_event
        log_security_event(
            logger, "METADATA_VALIDATION_FAILED", requester.get("user_id"),
            role=requester.get("role"), department=requester.get("department"),
            error=e.detail, source_name=req.source_name
        )
        raise

    try:
        # Use original working service
        document_manager: DocumentManager = get_document_instance()
        result = await document_manager.add_document_to_rag_local(
            source_name=req.source_name,
            text=req.text,
            metadata=metadata,
            created_by=requester.get("user_id")
        )

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


@router.post("/documents/add-file", response_model=AddResponse,
             dependencies=[Depends(require_roles(EMPLOYEE_PLUS_ROLES))])
async def add_document_file(
        file: UploadFile = File(...),
        requester: Dict[str, Any] = Depends(get_current_user),
        department: Optional[str] = DEFAULT_DEPARTMENT,
        sensitivity: Optional[str] = DEFAULT_SENSITIVITY
):
    """Upload and add document file - original working implementation."""
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail=HTTP_MESSAGES["FILE_EMPTY"])
    if len(raw) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail=HTTP_MESSAGES["FILE_TOO_LARGE"])

    try:
        text_content = raw.decode("utf-8", errors="ignore")
    except Exception as e:
        logger.exception("Failed to decode uploaded file: %s", e)
        raise HTTPException(status_code=400, detail=HTTP_MESSAGES["FILE_DECODE_ERROR"])

    # Determine format from extension
    ext = os.path.splitext(file.filename)[1].lower()
    if ext in MARKDOWN_EXTENSIONS:
        fmt = RawFormat.MARKDOWN
    elif ext in HTML_EXTENSIONS:
        fmt = RawFormat.HTML
    elif ext in JSON_EXTENSIONS:
        fmt = RawFormat.JSON
    else:
        fmt = RawFormat.PLAIN

    try:
        text = parse_text(text_content, format=fmt)
    except Exception as e:
        logger.warning(f"Failed to parse file {file.filename} as {fmt}, falling back to plain text. Error: {e}")
        text = text_content

    metadata = {"department": department, "sensitivity": sensitivity, "ingested_by": requester.get("user_id")}

    try:
        validate_metadata(metadata, requester)
    except HTTPException as e:
        from app.logging_config import log_security_event
        log_security_event(
            logger, "FILE_UPLOAD_VALIDATION_FAILED", requester.get("user_id"),
            role=requester.get("role"), department=requester.get("department"),
            filename=file.filename, error=e.detail, file_size=len(raw)
        )
        raise

    try:
        document_manager: DocumentManager = get_document_instance()
        result = await document_manager.add_document_to_rag_local(
            source_name=file.filename,
            text=text,
            metadata=metadata,
            created_by=requester.get("user_id")
        )

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


@router.post("/documents/seed", response_model=AddResponse,
             dependencies=[Depends(require_roles(SUPER_ADMIN_ROLES))])
async def seed_defaults(
        requester: Dict[str, Any] = Depends(get_current_user),
        reseed: bool = False
):
    """Seed default documents - original working implementation."""
    try:
        # Use original working service
        document_manager: DocumentManager = get_document_instance()
        ids = await document_manager.seed_from_file(force_reseed=reseed)
        if ids:
            return AddResponse(message=f"Seeded default docs from companyData. Chunks added: {len(ids)}.",
                               chunk_count=len(ids))
        else:
            msg = "Seed operation skipped: collection not empty (use ?reseed=true to force) or no files found."
            return AddResponse(message=msg, chunk_count=0)

    except Exception as e:
        logger.exception("Seeding failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/clear", response_model=AddResponse,
             dependencies=[Depends(require_roles(SUPER_ADMIN_ROLES))])
def clear_store(
        requester: Dict[str, Any] = Depends(get_current_user)
) -> AddResponse:
    """Clear document store - original working implementation."""
    try:
        # Use original working service
        document_manager: DocumentManager = get_document_instance()
        document_manager.clear_collection()
        return AddResponse(message=HTTP_MESSAGES["COLLECTION_CLEARED"], chunk_count=0)
    except Exception as e:
        logger.exception("Failed to clear collection: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# Document Versioning (use original working services)
# ---------------------------
@router.post("/documents/update", response_model=UpdateDocumentResponse,
             dependencies=[Depends(require_roles(EMPLOYEE_PLUS_ROLES))])
async def update_document(
        req: UpdateDocumentRequest,
        requester: Dict[str, Any] = Depends(get_current_user)
):
    """Update document - original working implementation."""
    try:
        # Use original working services
        container = get_container()
        container.initialize()
        version_manager = container.get_version_manager()

        latest_version = version_manager.get_latest_version(req.document_id)
        if not latest_version:
            raise HTTPException(status_code=404, detail=f"Document {req.document_id} not found")

        current_metadata = latest_version.get("metadata", {})
        current_dept = current_metadata.get("department", "General")
        user_role = requester.get("role")
        user_dept = requester.get("department")

        # Department ownership check (unless high-level roles)
        user_level = ROLE_LEVELS.get(user_role, 0)
        if user_level < HR_LEVEL_THRESHOLD:
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
            updated_metadata = {**current_metadata, **req.metadata}
            validate_metadata(updated_metadata, requester)

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
            updated_metadata = current_metadata

        # Perform update using original service
        document_manager: DocumentManager = get_document_instance()
        result = await document_manager.update_document_version(
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
            chunk_count=result["chunk_count"],
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
            dependencies=[Depends(require_roles(EMPLOYEE_PLUS_ROLES))])
async def list_all_documents(
        department: Optional[str] = None,
        status: Optional[str] = None,
        latest_only: bool = True
):
    """List all documents - original working implementation."""
    try:
        # Use original working service
        document_manager: DocumentManager = get_document_instance()
        documents = await document_manager.list_documents(
            department=department,
            status=status,
            latest_only=latest_only
        )
        return DocumentListResponse(documents=documents, count=len(documents))
    except Exception as e:
        logger.exception("Failed to list documents: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/versions", response_model=VersionHistoryResponse,
            dependencies=[Depends(require_roles(EMPLOYEE_PLUS_ROLES))])
async def get_version_history(document_id: str):
    """Get version history - original working implementation."""
    try:
        container = get_container()
        container.initialize()
        version_manager = container.get_version_manager()
        versions = version_manager.get_version_history(document_id)
        return VersionHistoryResponse(document_id=document_id, versions=versions)
    except Exception as e:
        logger.exception("Failed to get version history: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/versions/{version}", response_model=DocumentVersionResponse,
            dependencies=[Depends(require_roles(EMPLOYEE_PLUS_ROLES))])
async def get_specific_version(document_id: str, version: str):
    """Get specific version - original working implementation."""
    try:
        document_manager: DocumentManager = get_document_instance()
        result = await document_manager.get_document_version(document_id, version)
        if not result:
            raise HTTPException(status_code=404, detail=f"Version {version} of document {document_id} not found")

        return DocumentVersionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to get document version: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/{document_id}/compare", response_model=CompareVersionsResponse,
            dependencies=[Depends(require_roles(EMPLOYEE_PLUS_ROLES))])
async def compare_versions(document_id: str, version1: str, version2: str):
    """Compare versions - original working implementation."""
    try:
        document_manager: DocumentManager = get_document_instance()
        result = await document_manager.compare_document_versions(document_id, version1, version2)
        if not result:
            raise HTTPException(status_code=404, detail="One or both versions not found")

        return CompareVersionsResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to compare versions: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/{document_id}/archive", response_model=AddResponse,
             dependencies=[Depends(require_roles(HR_PLUS_ROLES))])
async def archive_version(document_id: str, version: str):
    """Archive version - original working implementation."""
    try:
        document_manager: DocumentManager = get_document_instance()
        success = await document_manager.archive_document_version(document_id, version)
        if not success:
            raise HTTPException(status_code=404, detail=f"Version {version} of document {document_id} not found")

        return AddResponse(message=f"Archived version {version} of document {document_id}", chunk_count=0)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Failed to archive version: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------
# Sentiment Analysis (use original working services)
# ---------------------------
@router.post("/sentiment", dependencies=[Depends(require_roles(SUPER_ADMIN_ROLES))])
def api_sentiment(req: SentimentRequest) -> Dict[str, Any]:
    """Analyze sentiment - original working implementation."""
    # from app.services.legacy.sentiment_classifier import get_global_sentiment
    # Sentiment classifier has been moved to modular architecture
    from app.modules.core.utils import analyze_sentiment
    result = analyze_sentiment(req.text)
    return {"ok": True, "result": result}


@router.get("/sentiment/stats", dependencies=[Depends(require_roles(SUPER_ADMIN_ROLES))])
def sentiment_stats_api() -> Dict[str, Any]:
    """Get sentiment stats - original working implementation."""
    # Get modular services for user management
    container = get_container()
    container.initialize()
    user_manager = container.get_user_manager()

    session_manager: ISessionManager = container.get_session_manager()
    return session_manager.get_sentiment_stats()


# ---------------------------
# System Status (use original working services)
# ---------------------------
@router.get("/embedding/status", dependencies=[Depends(require_roles(MANAGER_PLUS_ROLES))])
def embedding_model_status():
    """Get embedding model status - refactored to use EmbeddingManager."""
    from app.modules.vector_db.embedding_manager import EmbeddingManager

    try:
        embedding_manager = EmbeddingManager()
        info = embedding_manager.get_model_info()
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