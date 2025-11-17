from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Logging setup
from app.logging_config import setup_logging

# LLM service (models / Pydantic models used for responses)
from app.services import llm_service
from app.services.llm_service import GenerationResponse

# RAG services:
# - rag_local_service: low-level local RAG, initialization, seeding, local add
# - rag_manual_service: simpler wrappers used by CLI/scripts and the ask-document endpoint
from app.services import rag_manual_service as rag_manual_service
from app.services import rag_local_service as rag_local_service

# Routers
from app.api_routes_local import router as local_router

logger = setup_logging()

# -----------------------------
# Lifespan Handler (startup/shutdown)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs on application startup and shutdown.
    - Initializes local RAG (embeddings + chroma)
    - Optionally seeds the DB from a default file (if present)
    """
    print("🔵 Application startup...")

    # Initialize the local RAG using rag_local_service
    try:
        if hasattr(rag_local_service, "initialize_local_rag"):
            rag_local_service.initialize_local_rag()
            print("✔ Local RAG initialized successfully.")
        else:
            print("⚠ initialize_local_rag() not found in rag_local_service.")
    except Exception as e:
        print(f"❌ Error initializing Local RAG: {e}")

    # Optional: Seed data at startup (uses seed_from_file in rag_local_service)
    try:
        if hasattr(rag_local_service, "seed_from_file"):
            seeded_ids = rag_local_service.seed_from_file()
            if seeded_ids:
                print(f"✔ Seeded default file. Chunks added: {len(seeded_ids)}")
            else:
                print("ℹ No seed file found, skipping startup seed.")
        else:
            print("ℹ seed_from_file() not found in rag_local_service; skipping seeding.")
    except Exception as e:
        print(f"⚠ Seeding at startup skipped or failed: {e}")

    yield

    print("🔴 Application shutdown...")


# -----------------------------
# Create FastAPI app
# -----------------------------
app = FastAPI(
    title="AI Engineering API",
    description="A foundational API for AI engineering skills development.",
    version="1.0.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(local_router)

# CORS (for development only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Health endpoint ---
@app.get("/", tags=["General"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Welcome to the AI Engineering API!"}


# --- LLM Endpoints (unchanged, forwarding to llm_service) ---
@app.post("/summarize",
          response_model=llm_service.SummarizationResponse,
          tags=["LLM Services"])
def summarize(request: llm_service.TextRequest):
    try:
        return llm_service.summarize_text(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate",
          response_model=llm_service.GenerationResponse,
          tags=["LLM Services"])
def generate(request: llm_service.TextRequest):
    try:
        return llm_service.generate_text(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/sentiment",
          response_model=llm_service.SentimentResponse,
          tags=["LLM Services"])
def sentiment(request: llm_service.TextRequest):
    try:
        return llm_service.classify_sentiment(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/openai",
          response_model=llm_service.GenerationResponse,
          tags=["LLM Services (OpenAI)"])
def generate_openai(request: llm_service.TextRequest):
    try:
        return llm_service.generate_text_openai(request)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/hf",
          response_model=llm_service.GenerationResponse,
          tags=["LLM Services (Hugging Face API)"])
def generate_hf(request: llm_service.TextRequest):
    try:
        return llm_service.generate_text_hf_inference_langchain(request)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/ideas",
          response_model=llm_service.IdeaResponse,
          tags=["LLM Services (LangChain)"])
def generate_ideas(request: llm_service.IdeaRequest):
    try:
        return llm_service.generate_content_ideas(request)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat",
          response_model=llm_service.ChatResponse,
          tags=["LLM Services (Conversational)"])
def chat(request: llm_service.ChatRequest):
    try:
        return llm_service.get_chat_response(request)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- RAG / Document endpoints ---

@app.post("/ask-document",
          response_model=GenerationResponse,
          tags=["RAG Services"])
def ask_document(request: llm_service.TextRequest):
    """
    Ask a question against documents using the manual RAG helper.
    Uses rag_manual_service.query_manual_rag which expects a plain text query.
    """
    try:
        # Query the manual RAG helper; it returns a dict including 'answer' when use_llm used,
        # or visible documents and public_summaries otherwise.
        result = rag_manual_service.query_manual_rag(query_text=request.text, n_results=3, requester=None, use_llm=False)
        # Prefer a generated answer if present, otherwise provide safe composed text.
        answer = result.get("answer") or result.get("context") or (result.get("public_summaries") and "\n\n".join(result.get("public_summaries"))) or "No relevant documents found."
        return GenerationResponse(generated_text=answer)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/add-document",
          response_model=GenerationResponse,
          tags=["RAG Services"])
async def add_document(file: UploadFile = File(...)):
    """
    Add a document using the manual helper (simpler ingestion).
    This endpoint is for quick ad-hoc uploads not via the local RAG endpoint.
    """
    try:
        content = await file.read()
        document_text = content.decode("utf-8", errors="ignore")
        # Use manual wrapper to add doc; it returns list of chunk ids
        ids = rag_manual_service.add_document_manual(source_name=file.filename, text=document_text, metadata=None)
        if ids:
            message = f"Successfully processed and added '{file.filename}'. {len(ids)} chunks ingested."
            return GenerationResponse(generated_text=message)
        else:
            raise HTTPException(status_code=400, detail="The file was empty or could not be processed.")
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")


# Local ingestion endpoint (uses the local rag service which is optimized for persistent storage)
from app.services.rag_local_service import add_document_to_rag_local

@app.post("/add-document-local",
          response_model=GenerationResponse,
          tags=["Local RAG"])
async def add_document_local(file: UploadFile = File(...)):
    """
    Add a new text document to the local RAG knowledge base (persistent).
    Uses local embeddings + chroma storage.
    """
    try:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        document_text = raw.decode("utf-8", errors="ignore")
        chunk_ids = add_document_to_rag_local(source_name=file.filename, text=document_text)
        if chunk_ids:
            msg = f"Successfully ingested '{file.filename}'. {len(chunk_ids)} chunks created and stored."
            return GenerationResponse(generated_text=msg)
        else:
            raise HTTPException(status_code=400, detail="Failed to process file or file was empty.")
    except HTTPException:
        raise
    except ConnectionError as ce:
        raise HTTPException(status_code=503, detail=str(ce))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")
