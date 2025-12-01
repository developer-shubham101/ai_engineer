from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Logging setup
from app.logging_config import setup_logging

# New modular architecture
from app.modules.integration import get_container

# Legacy routers (will be updated to use modular architecture)
from app.api_routes_rag import router as rag_router
from app.api_routes_auth import router as auth_router

logger = setup_logging()

# -----------------------------
# Lifespan Handler (startup/shutdown)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
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

    # Legacy initialization (for backward compatibility)
    try:
        from app.services import rag_local_service
        if hasattr(rag_local_service, "initialize_local_rag"):
            rag_local_service.initialize_local_rag()
            logger.info("Legacy RAG initialized successfully.")
    except Exception as e:
        logger.warning(f"Legacy RAG initialization failed: {e}")

    # Legacy user database
    try:
        from app.services.user_service import init_user_db
        init_user_db(reset_on_start=False)
        logger.info("Legacy user database initialized.")
    except Exception as e:
        logger.warning(f"Legacy user database initialization failed: {e}")

    # Legacy version tracking
    try:
        from app.services.version_tracking import init_version_db
        init_version_db(reset_on_start=False)
        logger.info("Legacy version tracking initialized.")
    except Exception as e:
        logger.warning(f"Legacy version tracking initialization failed: {e}")

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
# app.include_router(training_router)

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


# Health endpoint
@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "architecture": "modular"}

# Test modular architecture endpoint
@app.get("/api/modules/status", tags=["Modules"])
def modules_status():
    """Check modular architecture status."""
    try:
        container = get_container()
        available_providers = container.get_rag_orchestrator().get_available_providers()
        return {
            "status": "initialized",
            "available_providers": available_providers,
            "modules": {
                "auth": "available",
                "vector_db": "available", 
                "llm": "available",
                "core": "available"
            }
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}

