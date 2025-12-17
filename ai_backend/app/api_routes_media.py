"""Media file serving API routes."""

import os
import logging
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import FileResponse
from typing import Dict, Any

from app.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/media", tags=["Media"])


@router.get("/{user_id}/{filename}")
async def serve_media_file(
    user_id: str,
    filename: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Serve media files with RBAC."""
    try:
        # Security: Users can only access their own files
        if current_user["user_id"] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        file_path = os.path.join("user_uploaded_files", user_id, filename)
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        # Determine media type
        if filename.endswith(('.mp3', '.wav', '.ogg')):
            media_type = "audio/mpeg"
        elif filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            media_type = "image/jpeg"
        elif filename.endswith('.pdf'):
            media_type = "application/pdf"
        else:
            media_type = "application/octet-stream"
        
        return FileResponse(
            path=file_path,
            media_type=media_type,
            filename=filename
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Media serving failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))