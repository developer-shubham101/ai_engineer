"""
Metadata Models for LLM-Assisted Document Enrichment

This module defines the data models for strict (system-controlled) and soft (LLM-generated)
metadata used in the document cleanup and enrichment pipeline.
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class StrictMetadata(BaseModel):
    """
    System-controlled metadata fields.
    These are never modified by LLM and are used for hard filtering.
    """
    document_type: str = Field(..., description="Type of document (e.g., memo, policy, report)")
    department: str = Field(..., description="Department owning the document")
    sensitivity: str = Field(..., description="Sensitivity level (public, confidential, etc.)")
    source: str = Field(..., description="Source file name or identifier")
    domain: Optional[str] = Field(None, description="Business domain")
    published_year: Optional[int] = Field(None, description="Year of publication")
    effective_date: Optional[str] = Field(None, description="Effective date of document")
    allowed_roles: Optional[List[str]] = Field(None, description="Roles allowed to access")
    tags: Optional[List[str]] = Field(default_factory=list, description="Manual tags")


class SoftMetadata(BaseModel):
    """
    LLM-generated metadata fields.
    These are used for ranking, context, and semantic search.
    """
    summary: str = Field(..., description="2-3 sentence summary of document content")
    keywords: List[str] = Field(..., description="5-10 relevant keywords extracted from content")
    themes: List[str] = Field(..., description="3-5 main themes or topics")
    entities: Optional[Dict[str, List[str]]] = Field(
        default_factory=dict,
        description="Named entities (people, organizations, locations)"
    )
    generated_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp when metadata was generated"
    )
    llm_model: Optional[str] = Field(None, description="LLM model used for generation")
    confidence: Optional[float] = Field(None, description="Confidence score (0-1)")


class EnrichedMetadata(BaseModel):
    """
    Combined metadata containing both strict and soft fields.
    This is the final metadata stored with cleaned documents.
    """
    # Strict metadata (preserved from original)
    strict: StrictMetadata
    
    # Soft metadata (LLM-generated)
    soft: SoftMetadata
    
    # Processing metadata
    enriched_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Timestamp when enrichment was completed"
    )
    processing_time_ms: Optional[float] = Field(None, description="Time taken to enrich (ms)")


class DocumentCleanupStatus(BaseModel):
    """
    Status of a single document in the cleanup pipeline.
    """
    source_path: str
    document_id: str
    status: str = Field(..., description="success, failed, skipped")
    error_message: Optional[str] = None
    processing_time_ms: Optional[float] = None
    enriched_path: Optional[str] = None


class CleanupReport(BaseModel):
    """
    Report of cleanup pipeline execution.
    """
    started_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Pipeline start timestamp"
    )
    completed_at: Optional[str] = Field(None, description="Pipeline completion timestamp")
    status: str = Field("running", description="running, completed, failed")
    
    # Statistics
    total_documents: int = 0
    processed_documents: int = 0
    successful_documents: int = 0
    failed_documents: int = 0
    skipped_documents: int = 0
    
    # Details
    document_statuses: List[DocumentCleanupStatus] = Field(default_factory=list)
    
    # Performance
    total_processing_time_ms: Optional[float] = None
    average_processing_time_ms: Optional[float] = None
    
    # Errors
    errors: List[str] = Field(default_factory=list)
    
    def mark_completed(self):
        """Mark the report as completed and calculate final statistics."""
        self.completed_at = datetime.utcnow().isoformat()
        self.status = "completed"
        
        if self.processed_documents > 0:
            total_time = sum(
                doc.processing_time_ms or 0 
                for doc in self.document_statuses 
                if doc.processing_time_ms
            )
            self.total_processing_time_ms = total_time
            self.average_processing_time_ms = total_time / self.processed_documents
    
    def add_document_status(self, status: DocumentCleanupStatus):
        """Add a document status to the report."""
        self.document_statuses.append(status)
        self.processed_documents += 1
        
        if status.status == "success":
            self.successful_documents += 1
        elif status.status == "failed":
            self.failed_documents += 1
            if status.error_message:
                self.errors.append(f"{status.document_id}: {status.error_message}")
        elif status.status == "skipped":
            self.skipped_documents += 1


class MetadataPreview(BaseModel):
    """
    Preview of original vs enriched metadata for comparison.
    """
    document_id: str
    source_path: str
    original_metadata: Dict[str, Any]
    enriched_metadata: Optional[EnrichedMetadata] = None
    has_enriched: bool = False
