from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Logging setup
from app.logging_config import setup_logging

# Routers
from app.api_routes_rag import router as rag_router
from app.api_routes_models import router as models_router
from app.api_routes_training import router as training_router

# Services
from app.services.local_model_manager import LocalModelManager
from app.services.model_manager import ModelManager
from app.services.model_training_service import ModelTrainingService

logger = setup_logging()

# Global instances
local_model_manager = None
model_manager = None
model_training_service = None

# -----------------------------
# Lifespan Handler (startup/shutdown)
# -----------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for LoRA training system
    """
    global local_model_manager, model_manager, model_training_service
    
    logger.info("Starting LoRA Training System...")
    
    try:
        # Initialize model managers
        logger.info("Initializing model managers...")
        local_model_manager = LocalModelManager()
        model_manager = ModelManager()
        
        # Initialize model training service
        logger.info("Initializing model training service...")
        model_training_service = ModelTrainingService()
        
        # Store instances in app state
        app.state.local_model_manager = local_model_manager
        app.state.model_manager = model_manager
        app.state.model_training_service = model_training_service
        
        logger.info("System initialization completed successfully!")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise
    finally:
        logger.info("Shutting down LoRA Training System...")

# -----------------------------
# Create FastAPI app
# -----------------------------
app = FastAPI(
    title="LoRA Training and Testing System",
    description="Minimal system for LoRA fine-tuning and model testing",
    version="1.0.0",
    lifespan=lifespan,
)

# Register routers
app.include_router(models_router, prefix="/api/models", tags=["Models"])
app.include_router(training_router, prefix="/api/training", tags=["Training"])
app.include_router(rag_router, prefix="/api", tags=["Query"])

# CORS (for development only)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Endpoints ---
@app.get("/", tags=["General"])
def read_root():
    """Root endpoint with system information"""
    return {
        "message": "LoRA Training and Testing System",
        "version": "1.0.0",
        "status": "operational",
        "features": [
            "LoRA fine-tuning",
            "Model testing",
            "Model management"
        ]
    }

@app.get("/health", tags=["General"])
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "components": {
            "local_models": "available",
            "training_service": "ready"
        }
    }

