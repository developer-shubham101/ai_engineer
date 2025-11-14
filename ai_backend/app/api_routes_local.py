from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path

from app.services.rag_local_service import (
    add_document_to_rag_local,
    query_local_rag,
    seed_from_file,
    clear_collection,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/local", tags=["Local RAG"])

# ------ request/response models ------
class AddDocRequest(BaseModel):
    text: str
    source_name: Optional[str] = "manual_upload"
    metadata: Optional[Dict[str, Any]] = None

class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    use_llm: bool = True
    max_tokens: int = 256

class RetrievedDoc(BaseModel):
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None
    distance: Optional[float] = None

class QueryResponse(BaseModel):
    answer: Optional[str] = None
    retrieved: List[RetrievedDoc] = []
    context: Optional[str] = None

class AddResponse(BaseModel):
    generated_text: str

# ------ endpoints ------
@router.get("/health")
async def health():
    return {"status": "ok", "service": "local_rag"}

@router.post("/add", response_model=AddResponse)
async def add_document_json(req: AddDocRequest):
    try:
        chunks = []
        # split naive if text long; rag_local_service does splitting if chunks None
        chunk_ids = add_document_to_rag_local(source_name=req.source_name, text=req.text, metadata=req.metadata)
        if chunk_ids:
            msg = f"Added document '{req.source_name}' with {len(chunk_ids)} chunks."
            return AddResponse(generated_text=msg)
        else:
            raise HTTPException(status_code=400, detail="No chunks were created.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Add document failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-file", response_model=AddResponse)
async def add_document_file(file: UploadFile = File(...)):
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="File empty.")
        text = raw.decode("utf-8", errors="ignore")
        chunk_ids = add_document_to_rag_local(source_name=file.filename, text=text)
        if chunk_ids:
            return AddResponse(generated_text=f"Added file {file.filename} ({len(chunk_ids)} chunks).")
        else:
            raise HTTPException(status_code=400, detail="No chunks added.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("File add failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query", response_model=QueryResponse)
async def query_local(req: QueryRequest):
    try:
        res = query_local_rag(query_text=req.question, n_results=req.top_k, use_llm=req.use_llm, max_tokens=req.max_tokens)
        retrieved = []
        # res contains documents, metadatas, ids, distances, context, maybe answer
        docs = res.get("documents", [])
        metadatas = res.get("metadatas", [])
        ids = res.get("ids", [])
        distances = res.get("distances", [])
        for i, doc in enumerate(docs):
            retrieved.append(RetrievedDoc(
                id=ids[i] if i < len(ids) else f"unknown_{i}",
                text=doc,
                metadata=metadatas[i] if i < len(metadatas) else None,
                distance=(distances[i] if i < len(distances) else None)
            ))
        return QueryResponse(answer=res.get("answer"), retrieved=retrieved, context=res.get("context"))
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reset")
async def reset_collection():
    try:
        clear_collection()
        return {"status": "reset"}
    except Exception as e:
        logger.exception("Reset failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/seed-default")
async def seed_default():
    try:
        ids = seed_from_file()
        return {"seeded_ids": ids}
    except Exception as e:
        logger.exception("Seeding failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
