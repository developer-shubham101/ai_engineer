# app/api_routes_training.py
"""
API routes for model training functionality.
"""

import logging
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from pydantic import BaseModel, Field

from app.dependencies import get_current_user, require_roles
from app.services.model_training_service import (
    train_company_model, 
    get_trained_models, 
    is_training_available
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/training", tags=["Model Training"])

# Request/Response models
class TrainingRequest(BaseModel):
    output_name: str = Field(default="llama-3.2-1b-company-tuned", description="Name for the trained model")
    max_samples: int = Field(default=1000, ge=100, le=5000, description="Maximum training samples")
    epochs: int = Field(default=3, ge=1, le=10, description="Number of training epochs")
    learning_rate: float = Field(default=2e-5, gt=0, lt=1, description="Learning rate")

class TrainingResponse(BaseModel):
    message: str
    training_id: str
    status: str

class ModelListResponse(BaseModel):
    models: List[Dict[str, Any]]
    count: int

class TrainingStatusResponse(BaseModel):
    available: bool
    message: str
    requirements: List[str]

# Training status tracking
_training_jobs = {}

@router.get("/status", response_model=TrainingStatusResponse)
async def get_training_status():
    """Check if training is available and get requirements."""
    available = is_training_available()
    
    if available:
        return TrainingStatusResponse(
            available=True,
            message="Training is available",
            requirements=[]
        )
    else:
        return TrainingStatusResponse(
            available=False,
            message="Training dependencies not installed",
            requirements=[
                "pip install transformers",
                "pip install datasets", 
                "pip install torch",
                "pip install accelerate"
            ]
        )

@router.post("/start", response_model=TrainingResponse, 
             dependencies=[Depends(require_roles(["SuperAdmin"]))])
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    requester: Dict[str, Any] = Depends(get_current_user)
):
    """Start training a model on company data."""
    if not is_training_available():
        raise HTTPException(
            status_code=400,
            detail="Training dependencies not available. Install: pip install transformers datasets torch"
        )
    
    # Generate training ID
    import uuid
    training_id = str(uuid.uuid4())[:8]
    
    # Track training job
    _training_jobs[training_id] = {
        "status": "starting",
        "started_by": requester.get("user_id"),
        "started_at": None,
        "completed_at": None,
        "error": None,
        "result": None
    }
    
    # Start training in background
    background_tasks.add_task(
        _run_training_job,
        training_id,
        request.output_name,
        request.max_samples,
        request.epochs,
        request.learning_rate
    )
    
    logger.info(
        "TRAINING_STARTED: user=%s training_id=%s model=%s samples=%d epochs=%d",
        requester.get("user_id"), training_id, request.output_name, 
        request.max_samples, request.epochs
    )
    
    return TrainingResponse(
        message=f"Training started for model '{request.output_name}'",
        training_id=training_id,
        status="starting"
    )

@router.get("/jobs/{training_id}")
async def get_training_job(
    training_id: str,
    requester: Dict[str, Any] = Depends(get_current_user)
):
    """Get status of a training job."""
    if training_id not in _training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = _training_jobs[training_id]
    
    # Only allow access to own jobs or SuperAdmin
    if (job["started_by"] != requester.get("user_id") and 
        requester.get("role") != "SuperAdmin"):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return job

@router.get("/models", response_model=ModelListResponse,
            dependencies=[Depends(require_roles(["SuperAdmin", "Manager"]))])
async def list_trained_models():
    """List all trained models."""
    try:
        models = get_trained_models()
        return ModelListResponse(models=models, count=len(models))
    except Exception as e:
        logger.exception("Failed to list trained models: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/models/{model_name}",
               dependencies=[Depends(require_roles(["SuperAdmin"]))])
async def delete_trained_model(
    model_name: str,
    requester: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a trained model."""
    try:
        from pathlib import Path
        models_dir = Path("models")
        
        # Remove model directory
        model_dir = models_dir / model_name
        if model_dir.exists():
            import shutil
            shutil.rmtree(model_dir)
        
        # Remove GGUF file
        gguf_file = models_dir / f"{model_name}.gguf"
        if gguf_file.exists():
            gguf_file.unlink()
        
        # Remove info file
        info_file = models_dir / f"{model_name}.json"
        if info_file.exists():
            info_file.unlink()
        
        logger.info(
            "MODEL_DELETED: user=%s model=%s",
            requester.get("user_id"), model_name
        )
        
        return {"message": f"Model '{model_name}' deleted successfully"}
        
    except Exception as e:
        logger.exception("Failed to delete model: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

async def _run_training_job(
    training_id: str,
    output_name: str,
    max_samples: int,
    epochs: int,
    learning_rate: float
):
    """Background task to run training."""
    from datetime import datetime
    
    job = _training_jobs[training_id]
    
    try:
        job["status"] = "running"
        job["started_at"] = datetime.utcnow().isoformat()
        
        # Run training
        result = await train_company_model(
            output_name=output_name,
            max_samples=max_samples,
            epochs=epochs,
            learning_rate=learning_rate
        )
        
        job["status"] = "completed"
        job["completed_at"] = datetime.utcnow().isoformat()
        job["result"] = result
        
        logger.info("TRAINING_COMPLETED: training_id=%s model=%s", training_id, output_name)
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.utcnow().isoformat()
        
        logger.exception("TRAINING_FAILED: training_id=%s error=%s", training_id, e)