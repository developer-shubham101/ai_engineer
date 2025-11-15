# app/api_routes_local.py

from typing import List, Optional, Dict, Any
import logging

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Header
from pydantic import BaseModel, Field


# --- new imports for auth/validation ---
from fastapi import Depends, Header
from enum import Enum
from datetime import datetime
import uuid

# import auth helper (simple API-key mapping)
from app.services.auth import get_user_from_api_key  # expects a small mapping in app/services/auth.py

from app.services.rag_local_service import (
    add_document_to_rag_local,
    query_local_rag,
    seed_from_file,
    clear_collection,
    update_metadata as rag_update_metadata,  # newly added helper
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
    # Human-readable message about the operation
    message: str
    # Number of chunks created (if any)
    chunk_count: int = 0
    # Optional: list of created chunk ids
    ids: Optional[List[str]] = None

# ---------------------------
# New: Sensitivity Enum & metadata helpers
# ---------------------------
class Sensitivity(str, Enum):
    # Public internal — visible to all authenticated users
    PUBLIC_INTERNAL = "public_internal"
    # Department confidential — visible to same department, HR, Legal, Executives
    DEPARTMENT_CONFIDENTIAL = "department_confidential"
    # Role confidential — visible to allowed roles and admins (HR/Legal/Exec)
    ROLE_CONFIDENTIAL = "role_confidential"
    # Highly confidential — visible only to Legal and Executives
    HIGHLY_CONFIDENTIAL = "highly_confidential"
    # Personal — owner only + HR/Legal/Executive
    PERSONAL = "personal"

ALLOWED_SENSITIVITY = {s.value for s in Sensitivity}

def get_requester(x_api_key: Optional[str] = Header(None)):
    """
    Simple learning-mode auth: maps API key header to a user dict.
    If missing/unknown, returns Guest (minimal access).
    """
    user = get_user_from_api_key(x_api_key) if x_api_key else None
    if not user:
        return {"user_id": None, "role": "Guest", "department": None}
    return user

def _enforce_sensitivity_creation(requester: Dict[str, Any], sensitivity: str, department: Optional[str] = None, owner_id: Optional[str] = None):
    """
    Enforce who can create a document with a given sensitivity.
    Returns None if allowed, otherwise raises HTTPException.
    Rules (configured for this demo):
      - public_internal: anyone
      - department_confidential: same department OR HR/Legal/Executive
      - role_confidential: allowed_roles values must be composed by admins/managers; allow HR/Legal/Executive
      - highly_confidential: only Legal or Executive
      - personal: require owner_id and must be HR or the owner or Legal/Executive
    """
    role = requester.get("role")
    user_dept = requester.get("department")
    if sensitivity == Sensitivity.PUBLIC_INTERNAL.value:
        return
    if sensitivity == Sensitivity.DEPARTMENT_CONFIDENTIAL.value:
        if role in ("HR", "Legal", "Executive"):
            return
        if user_dept and department and user_dept == department:
            return
        raise HTTPException(status_code=403, detail="Cannot create department_confidential docs outside your department.")
    if sensitivity == Sensitivity.ROLE_CONFIDENTIAL.value:
        if role in ("HR", "Legal", "Executive"):
            return
        # otherwise, disallow creation to avoid accidental leaking of allowed_roles control
        raise HTTPException(status_code=403, detail="Only HR/Legal/Executive can create role_confidential documents in this demo.")
    if sensitivity == Sensitivity.HIGHLY_CONFIDENTIAL.value:
        if role in ("Legal", "Executive"):
            return
        raise HTTPException(status_code=403, detail="Only Legal or Executive can create highly_confidential documents.")
    if sensitivity == Sensitivity.PERSONAL.value:
        if owner_id is None:
            raise HTTPException(status_code=400, detail="owner_id is required for personal documents.")
        if role in ("HR", "Legal", "Executive"):
            return
        if requester.get("user_id") == owner_id:
            return
        raise HTTPException(status_code=403, detail="Only HR, Legal, Executive or the owner may create personal documents.")
    # fallback
    raise HTTPException(status_code=400, detail="Invalid sensitivity value.")

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

    # visible (post-RBAC) docs
    retrieved_docs = res.get("documents") or []
    metadatas = res.get("metadatas") or []
    ids = res.get("ids") or []
    distances = res.get("distances") or []

    # raw (pre-RBAC) docs — useful to decide UX for users who had documents filtered
    raw_docs = res.get("raw_documents") or []
    raw_metas = res.get("raw_metadatas") or []
    raw_ids = res.get("raw_ids") or []

    # normalize possible nested lists (Chroma client variations)
    if retrieved_docs and isinstance(retrieved_docs[0], list):
        retrieved_docs = retrieved_docs[0]
    if metadatas and isinstance(metadatas[0], list):
        metadatas = metadatas[0]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    if distances and isinstance(distances[0], list):
        distances = distances[0]

    if raw_docs and isinstance(raw_docs[0], list):
        raw_docs = raw_docs[0]
    if raw_metas and isinstance(raw_metas[0], list):
        raw_metas = raw_metas[0]
    if raw_ids and isinstance(raw_ids[0], list):
        raw_ids = raw_ids[0]

    # build RetrievedDoc list from visible docs
    docs: List[RetrievedDoc] = []
    for i, doc_text in enumerate(retrieved_docs):
        meta = metadatas[i] if i < len(metadatas) else None
        id_ = ids[i] if i < len(ids) else f"doc_{i}"
        dist = distances[i] if i < len(distances) else None
        docs.append(RetrievedDoc(id=str(id_), text=doc_text, metadata=meta, distance=dist))

    # UX: determine what to show when user had docs filtered
    answer = res.get("answer")
    filtered_count = res.get("filtered_out_count", 0)
    public_summaries = res.get("public_summaries", []) or []
    filtered_details = res.get("filtered_details", []) if requester.get("role") in ("Executive", "Legal") else []

    # Friendly, informative fallback logic:
    if not answer:
        if docs:
            answer = "I found some relevant documents; see 'retrieved' for snippets and context."
        elif filtered_count > 0:
            if public_summaries:
                summary_text = "\n\n".join(public_summaries)
                answer = (
                    "I found restricted documents that are not visible to your role. "
                    "Here are safe, non-confidential summaries:\n\n" + summary_text +
                    "\n\nIf you need full access, please contact the document owner or HR."
                )
            else:
                answer = (
                    "I found documents relevant to your question, but they are restricted and not visible to your role. "
                    "If you believe you need access, please contact HR or Legal to request permission."
                )
        else:
            answer = "No relevant documents found in the knowledge base."

    resp = QueryResponse(answer=answer, retrieved=docs, context=res.get("context"))

    # attach admin-only debug info into response.context for Executive/Legal
    if filtered_details and isinstance(resp, QueryResponse):
        if requester.get("role") in ("Executive", "Legal"):
            resp.context = (resp.context or "") + "\n\n[ADMIN FILTERED DETAILS]\n" + str(filtered_details)

    return resp



@router.post("/add", response_model=AddResponse)
async def add_document_json(req: AddDocRequest, requester: Dict[str, Any] = Depends(get_requester)):
    """
    Add document via JSON payload. Metadata allowed:
      - department, sensitivity, tags, public_summary, owner_id, allowed_roles
    Enforces sensitivity creation rules based on requester role.
    """
    try:
        metadata = req.metadata or {}
        # supply defaults
        metadata.setdefault("department", metadata.get("department", "General"))
        metadata.setdefault("sensitivity", metadata.get("sensitivity", Sensitivity.PUBLIC_INTERNAL.value))
        # sanitize tags if list -> csv
        if isinstance(metadata.get("tags"), list):
            metadata["tags"] = ",".join(str(x) for x in metadata["tags"])
        # enforce allowed sensitivity values
        sens = metadata.get("sensitivity")
        if sens not in ALLOWED_SENSITIVITY:
            raise HTTPException(status_code=400, detail=f"Invalid sensitivity '{sens}'. Allowed: {list(ALLOWED_SENSITIVITY)}")
        # enforce creation permission
        _enforce_sensitivity_creation(requester, sens, department=metadata.get("department"), owner_id=metadata.get("owner_id"))

        # attach ingest metadata
        metadata["ingested_by"] = requester.get("user_id")
        metadata["ingested_at"] = datetime.utcnow().isoformat() + "Z"

        # call service
        chunk_ids = add_document_to_rag_local(source_name=req.source_name, text=req.text, metadata=metadata)
        if chunk_ids:
            msg = f"Added document '{req.source_name}' with {len(chunk_ids)} chunks."
            return AddResponse(message=msg, chunk_count=len(chunk_ids), ids=chunk_ids)
        else:
            raise HTTPException(status_code=400, detail="No chunks were created.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Add document failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-file", response_model=AddResponse)
async def add_document_file(
    file: UploadFile = File(...),
    requester: Dict[str, Any] = Depends(get_requester),
    department: Optional[str] = "General",
    sensitivity: Optional[str] = Sensitivity.PUBLIC_INTERNAL.value,
    tags: Optional[str] = None,
    public_summary: Optional[str] = None,
    owner_id: Optional[str] = None,
):
    """
    Upload a text file and attach metadata via query/form fields.
    Example: ?department=HR&sensitivity=department_confidential
    """
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="File empty.")
        if len(raw) > 5 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="File too large (max 5 MB).")
        text = raw.decode("utf-8", errors="ignore")

        metadata = {
            "department": department or "General",
            "sensitivity": sensitivity or Sensitivity.PUBLIC_INTERNAL.value,
            "tags": tags or "",
            "public_summary": public_summary,
            "owner_id": owner_id,
        }

        # Validate sensitivity value
        if metadata["sensitivity"] not in ALLOWED_SENSITIVITY:
            raise HTTPException(status_code=400, detail=f"Invalid sensitivity '{metadata['sensitivity']}'")

        # enforce creation permission
        _enforce_sensitivity_creation(requester, metadata["sensitivity"], department=metadata["department"], owner_id=owner_id)

        # attach ingest info
        metadata["ingested_by"] = requester.get("user_id")
        metadata["ingested_at"] = datetime.utcnow().isoformat() + "Z"

        chunk_ids = add_document_to_rag_local(source_name=file.filename, text=text, metadata=metadata)
        if chunk_ids:
            msg = f"Added file {file.filename} ({len(chunk_ids)} chunks)."
            return AddResponse(message=msg, chunk_count=len(chunk_ids), ids=chunk_ids)
        else:
            raise HTTPException(status_code=400, detail="No chunks added.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("File add failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


class UpdateMetadataRequest(BaseModel):
    ids: List[str]
    metadata: Dict[str, Any]

@router.post("/update-metadata")
async def update_metadata_endpoint(req: UpdateMetadataRequest, requester: Dict[str, Any] = Depends(get_requester)):
    """
    Update metadata for existing chunk ids. Restricted to HR/Legal/Executive/Manager in this demo.
    """
    role = requester.get("role")
    if role not in ("HR", "Legal", "Executive", "Manager"):
        raise HTTPException(status_code=403, detail="Only HR/Legal/Executive/Manager can update metadata in this demo.")

    # sanitize sensitivity if present
    meta = req.metadata
    sens = meta.get("sensitivity")
    if sens and sens not in ALLOWED_SENSITIVITY:
        raise HTTPException(status_code=400, detail=f"Invalid sensitivity '{sens}'")

    try:
        ok = rag_update_metadata(req.ids, meta)
        if ok:
            return {"message": "Metadata update attempted", "updated_ids": req.ids}
        else:
            raise HTTPException(status_code=500, detail="Metadata update failed")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Metadata update failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


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
