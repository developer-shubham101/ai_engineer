# app/api_routes_local.py

from typing import List, Optional, Dict, Any
import logging

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

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/local", tags=["Local RAG"])

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
async def query_local(req: QueryRequest, requester: Dict[str, Any] = Depends(get_requester)):
    """
    Query the local RAG. Role info is passed via 'requester' so the service can filter
    retrieved docs by sensitivity/department/allowed_roles.
    """
    logger.info("Query request: role=%s user=%s question=%s", requester.get("role"), requester.get("user_id"), req.question)

    try:
        res = query_local_rag(
            query_text=req.question,
            n_results=req.top_k,
            requester=requester,       # pass role info for filtering in service
            use_llm=req.use_llm,
            max_tokens=req.max_tokens,
        )
    except Exception as e:
        logger.exception("RAG query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

    docs = []
    # Normalize retrieved items if present in result
    retrieved_docs = res.get("documents") or []
    metadatas = res.get("metadatas") or []
    ids = res.get("ids") or []
    distances = res.get("distances") or []

    # some clients may return nested lists - try to handle gracefully
    if retrieved_docs and isinstance(retrieved_docs[0], list):
        retrieved_docs = retrieved_docs[0]
    if metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if distances and isinstance(distances[0], list):
        distances = distances[0]

    # build RetrievedDoc list
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
    metadata["ingested_at"] = metadata.get("ingested_at")  # optional if caller provided

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
