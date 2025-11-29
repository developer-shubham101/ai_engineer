from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Logging setup
from app.logging_config import setup_logging

# RAG services
from app.services import rag_local_service

# Routers
from app.api_routes_rag import router as rag_router
from app.api_routes_auth import router as auth_router
from app.api_routes_training import router as training_router

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

    # Initialize version tracking database
    try:
        from app.services.version_tracking import init_version_db
        init_version_db(reset_on_start=False)  # Set to True for development to reset versions
        logger.info("Version tracking database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing version tracking database: {e}")

    # Seed from default path (supports version folders: data/company/v1/, v2/, v3/...
    try:
        from pathlib import Path
        default_seed = Path.cwd() / "data" / "company"
        if default_seed.exists() and hasattr(rag_local_service, "seed_from_file"):
            logger.info("Attempting to seed from %s (version-aware)", default_seed)
            seeded_ids = await rag_local_service.seed_from_file(seed_path=default_seed, force_reseed=False)
            if seeded_ids:
                logger.info(f"Seeded {len(seeded_ids)} chunks from versioned folders")
            else:
                logger.info("No seed or collection already populated, skipping startup seed.")
        else:
            logger.info("Seed path not found or seed_from_file() not available; skipping seeding.")
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
app.include_router(training_router)

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


# Legacy endpoints removed - use /api/rag/{provider}/query instead
# All functionality moved to proper RAG router endpoints

