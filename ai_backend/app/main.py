from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Dict, Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api_routes_auth import router as auth_router
from app.api_routes_rag import router as rag_router
from app.api_routes_conversations import router as conversations_router
from app.api_routes_templates import router as templates_router
from app.api_routes_audio import router as audio_router
from app.api_routes_vision import router as vision_router
from app.api_routes_media import router as media_router
from app.logging_config import setup_logging
from app.modules.integration import get_container
from app.modules.config import API_PREFIX

logger = setup_logging()


# -----------------------------
# Lifespan Handler (startup/shutdown)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Initialize modular architecture and legacy systems.
    """
    logger.info("Application startup...")

    # Initialize modular architecture
    try:
        container = get_container()
        container.initialize()
        logger.info("Modular architecture initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing modular architecture: {e}")

    logger.info("Modular architecture startup complete.")

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
app.include_router(conversations_router)  # Conversation history routes
app.include_router(templates_router)  # Template management
app.include_router(audio_router)  # NEW: Audio processing routes
app.include_router(vision_router)  # NEW: Vision processing routes
app.include_router(media_router)  # NEW: Media serving routes
# app.include_router(training_router)

# CORS (for development only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Health endpoint ---
@app.get("/", tags=["General"])
def read_root() -> Dict[str, str]:
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Welcome to the AI Engineering API!"}


# Health endpoint
@app.get("/health", tags=["General"])
def health_check() -> Dict[str, str]:
    """Health check endpoint."""
    return {"status": "healthy", "architecture": "modular"}


# Test modular architecture endpoint
@app.get(f"{API_PREFIX}/modules/status", tags=["Modules"])
def modules_status() -> Dict[str, Any]:
    """Check modular architecture status."""
    try:
        container = get_container()
        return {
            "status": "initialized",
            "modules": {
                "auth": "available",
                "vector_db": "available",
                "llm": "available",
                "core": "available"
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}
