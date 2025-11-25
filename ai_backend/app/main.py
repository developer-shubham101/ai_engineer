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
from app.services import rag_local_service as rag_local_service

# Routers
from app.api_routes_rag import router as rag_router
from app.api_routes_auth import router as auth_router

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
    logger.info("Application startup...")

    # Initialize the local RAG using rag_local_service
    try:
        if hasattr(rag_local_service, "initialize_local_rag"):
            rag_local_service.initialize_local_rag()
            logger.info("Local RAG initialized successfully.")
        else:
            logger.warning("initialize_local_rag() not found in rag_local_service.")
    except Exception as e:
        logger.error(f"Error initializing Local RAG: {e}")

    # Initialize user database
    try:
        from app.services.user_service import init_user_db
        init_user_db(reset_on_start=False)  # Set to True for development to reset users
        logger.info("User database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing user database: {e}")

    # Optional: Seed data at startup (uses seed_from_file in rag_local_service)
    try:
        if hasattr(rag_local_service, "seed_from_file"):
            # This calls seed_from_file(force_reseed=False) implicitly
            seeded_ids = await rag_local_service.seed_from_file()
            if seeded_ids:
                logger.info(f"Seeded default file. Chunks added: {len(seeded_ids)}")
            else:
                logger.info("No seed file found or collection was already populated, skipping startup seed.")
        else:
            logger.info("seed_from_file() not found in rag_local_service; skipping seeding.")
    except Exception as e:
        logger.warning(f"Seeding at startup skipped or failed: {e}")

    yield

    logger.info("Application shutdown...")

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
app.include_router(auth_router)
app.include_router(rag_router)

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
async def ask_document(request: llm_service.TextRequest):
    """
    Ask a question against documents using the local RAG service.
    """
    try:
        # Query the local RAG service
        result = await rag_local_service.query_local_rag(query_text=request.text, n_results=3, requester=None, use_llm=False)
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
    Add a document using the local RAG service.
    """
    try:
        content = await file.read()
        document_text = content.decode("utf-8", errors="ignore")
        # Use local RAG service to add doc; it returns list of chunk ids
        ids = await rag_local_service.add_document_to_rag_local(source_name=file.filename, text=document_text, metadata=None)
        if ids:
            message = f"Successfully processed and added '{file.filename}'. {len(ids)} chunks ingested."
            return GenerationResponse(generated_text=message)
        else:
            raise HTTPException(status_code=400, detail="The file was empty or could not be processed.")
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process document: {e}")

