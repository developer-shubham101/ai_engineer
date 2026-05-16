"""
Cleanup API Routes

API endpoints for document cleanup and metadata enrichment pipeline.
"""
from __future__ import annotations

import logging
from typing import Optional, Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from app.modules.core.cleanup_service import CleanupService
from app.modules.core.metadata_models import CleanupReport
from app.dependencies import get_current_user
from app.modules.integration import get_container

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cleanupdata", tags=["Cleanup"])


# Global state for tracking cleanup status
_current_cleanup_report: Optional[CleanupReport] = None
_cleanup_in_progress: bool = False


# Request/Response Models
class CleanupStartRequest(BaseModel):
    """Request to start cleanup pipeline."""
    force: bool = Field(
        default=False, 
        description="Force cleanup even if one is already in progress"
    )


class CleanupStatusResponse(BaseModel):
    """Response for cleanup status check."""
    in_progress: bool
    report: Optional[CleanupReport] = None


class MetadataPreviewResponse(BaseModel):
    """Response for metadata preview."""
    document_id: str
    version: str
    source_path: str
    original_metadata: Dict[str, Any]
    enriched_metadata: Optional[Dict[str, Any]] = None
    has_enriched: bool


# Dependency to get cleanup service
def get_cleanup_service() -> CleanupService:
    """Get cleanup service from container."""
    container = get_container()
    
    # Get metadata generator
    metadata_generator = container.get_metadata_generator()
    
    # Create cleanup service
    return CleanupService(
        metadata_generator=metadata_generator,
        source_base_dir="data/company",
        output_base_dir="data/cleaned/company"
    )


@router.post("", response_model=CleanupReport)
async def start_cleanup(
    request: CleanupStartRequest = CleanupStartRequest(),
    requester: Dict[str, Any] = Depends(get_current_user),
    cleanup_service: CleanupService = Depends(get_cleanup_service)
):
    """
    Start the document cleanup and enrichment pipeline.
    
    Scans data/company/v* directories, processes all documents with LLM metadata generation,
    and saves enriched versions to cleaned/company/v*.
    
    Requires authentication.
    """
    global _current_cleanup_report, _cleanup_in_progress
    
    logger.info(f"Cleanup requested by user: {requester.get('user_id')}")
    
    # Check if cleanup is already in progress
    if _cleanup_in_progress and not request.force:
        raise HTTPException(
            status_code=409,
            detail="Cleanup is already in progress. Use force=true to override."
        )
    
    try:
        # Mark cleanup as in progress
        _cleanup_in_progress = True
        
        # Run cleanup pipeline
        logger.info("Starting cleanup pipeline...")
        report = await cleanup_service.cleanup_all_versions()
        
        # Store report
        _current_cleanup_report = report
        
        logger.info(
            f"Cleanup completed: {report.successful_documents}/{report.total_documents} "
            f"documents processed successfully"
        )
        
        return report
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {str(e)}"
        )
    finally:
        # Mark cleanup as complete
        _cleanup_in_progress = False


@router.get("/status", response_model=CleanupStatusResponse)
async def get_cleanup_status(
    requester: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get the status of the cleanup pipeline.
    
    Returns the current cleanup report if available, or the last completed report.
    
    Requires authentication.
    """
    global _current_cleanup_report, _cleanup_in_progress
    
    return CleanupStatusResponse(
        in_progress=_cleanup_in_progress,
        report=_current_cleanup_report
    )


@router.get("/preview/{document_id}", response_model=MetadataPreviewResponse)
async def preview_metadata(
    document_id: str,
    version: str = "v1",
    requester: Dict[str, Any] = Depends(get_current_user),
    cleanup_service: CleanupService = Depends(get_cleanup_service)
):
    """
    Preview original vs enriched metadata for a document.
    
    Useful for validating metadata quality before full cleanup.
    
    Args:
        document_id: Document identifier (filename without extension)
        version: Version directory (default: v1)
    
    Requires authentication.
    """
    logger.info(
        f"Metadata preview requested for {document_id} (version: {version}) "
        f"by user: {requester.get('user_id')}"
    )
    
    try:
        preview = await cleanup_service.preview_metadata(document_id, version)
        
        if not preview:
            raise HTTPException(
                status_code=404,
                detail=f"Document {document_id} not found in version {version}"
            )
        
        return MetadataPreviewResponse(**preview)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Preview failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Preview failed: {str(e)}"
        )
