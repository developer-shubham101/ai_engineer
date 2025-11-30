# app/api_routes_models.py
"""
Model management API endpoints.
"""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

# Simplified dependencies - no auth required for minimal system
def get_current_user():
    return {"user_id": "system", "role": "SuperAdmin"}

def require_roles(roles):
    def dependency():
        return get_current_user()
    return dependency
from app.services.local_model_manager import get_model_manager

router = APIRouter(prefix="/api/models", tags=["Models"])

class ModelInfo(BaseModel):
    key: str
    name: str
    size: str
    description: str
    available: bool
    path: str = None
    size_mb: float = None
    recommended: bool = False
    is_default: bool = False

class ModelListResponse(BaseModel):
    models: List[ModelInfo]
    default_model: str
    available_count: int
    total_count: int

class DownloadableModel(BaseModel):
    key: str
    name: str
    size: str
    description: str
    download_url: str
    gguf_file: str

class DownloadableModelsResponse(BaseModel):
    models: List[DownloadableModel]
    count: int

@router.get("/list", response_model=ModelListResponse)
async def list_models():
    """List all configured local models with availability status."""
    manager = get_model_manager()
    available_models = manager.get_available_models()
    default_model = manager.get_default_model()
    
    models = []
    available_count = 0
    
    for key, info in available_models.items():
        model_info = ModelInfo(
            key=key,
            name=info.get("name", key),
            size=info.get("size", "Unknown"),
            description=info.get("description", ""),
            available=info.get("available", False),
            path=info.get("path"),
            size_mb=info.get("size_mb"),
            recommended=info.get("recommended", False),
            is_default=(key == default_model)
        )
        models.append(model_info)
        
        if model_info.available:
            available_count += 1
    
    return ModelListResponse(
        models=models,
        default_model=default_model,
        available_count=available_count,
        total_count=len(models)
    )

@router.get("/downloadable", response_model=DownloadableModelsResponse)
async def list_downloadable_models():
    """List models that can be downloaded."""
    manager = get_model_manager()
    downloadable = manager.list_downloadable_models()
    
    models = [DownloadableModel(**model) for model in downloadable]
    
    return DownloadableModelsResponse(
        models=models,
        count=len(models)
    )

@router.get("/best", response_model=Dict[str, Any])
async def get_best_model():
    """Get the best available model for use."""
    manager = get_model_manager()
    best_model = manager.get_best_available_model()
    
    if not best_model:
        raise HTTPException(
            status_code=404,
            detail="No local models available. Please download a model first."
        )
    
    model_info = manager.get_model_info(best_model)
    return {
        "key": best_model,
        "info": model_info,
        "message": f"Best available model: {model_info.get('name', best_model)}"
    }

@router.post("/refresh")
async def refresh_models(
    requester: Dict[str, Any] = Depends(get_current_user)
):
    """Refresh the model cache (scan for new models)."""
    manager = get_model_manager()
    manager.refresh_cache()
    
    available_models = manager.get_available_models()
    available_count = sum(1 for info in available_models.values() if info.get("available"))
    
    return {
        "message": "Model cache refreshed",
        "available_models": available_count,
        "total_models": len(available_models)
    }