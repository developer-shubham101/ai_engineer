"""Vision processing API routes."""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from pydantic import BaseModel

from app.dependencies import get_current_user
from app.modules.multimodal import create_vision_provider, LocalFileManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/vision", tags=["Vision"])


class VisionResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    file_path: str = None
    error: str = None


@router.post("/ocr", response_model=VisionResponse)
async def extract_text_from_image(
    file: UploadFile = File(...),
    provider: str = "auto",
    conversation_id: str = "default",
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Extract text from image using OCR."""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be image format")
        
        # Save uploaded file
        file_manager = LocalFileManager()
        file_content = await file.read()
        file_path = await file_manager.save_uploaded_file(
            file_content, current_user["user_id"], "image", conversation_id
        )
        
        # Process with OCR
        vision_provider = create_vision_provider(provider)
        result = await vision_provider.extract_text(file_path)

        logger.info("OCR completed: success=%s provider=%s file_path=%s", result.success, provider, file_path)
        
        return VisionResponse(
            success=result.success,
            data=result.data,
            file_path=file_path or "",
            error=result.error or ""
        )
        
    except Exception as e:
        logger.exception("OCR processing failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/describe", response_model=VisionResponse)
async def describe_image(
    file: UploadFile = File(...),
    provider: str = "tesseract",
    conversation_id: str = "default",
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Generate description of image."""
    try:
        # Validate file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be image format")
        
        # Save uploaded file
        file_manager = LocalFileManager()
        file_content = await file.read()
        file_path = await file_manager.save_uploaded_file(
            file_content, current_user["user_id"], "image", conversation_id
        )
        
        # Process with vision provider
        vision_provider = create_vision_provider(provider)
        result = await vision_provider.describe_image(file_path)
        
        return VisionResponse(
            success=result.success,
            data=result.data,
            file_path=file_path,
            error=result.error
        )
        
    except Exception as e:
        logger.exception("Image description failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze", response_model=VisionResponse)
async def analyze_image(
    file: UploadFile = File(...),
    provider: str = "auto",
    conversation_id: str = "default",
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Analyze image (placeholder for future object detection)."""
    try:
        # For now, just return basic image info
        return await describe_image(file, provider, conversation_id, current_user)
        
    except Exception as e:
        logger.exception("Image analysis failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
