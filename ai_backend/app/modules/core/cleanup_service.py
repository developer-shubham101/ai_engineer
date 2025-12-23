"""
Document Cleanup Service

This module orchestrates the cleanup and enrichment pipeline for documents.
Scans directories, processes documents with LLM metadata generation, and saves enriched versions.
"""
from __future__ import annotations

import os
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from app.modules.core.metadata_models import (
    StrictMetadata,
    SoftMetadata,
    EnrichedMetadata,
    CleanupReport,
    DocumentCleanupStatus
)
from app.modules.core.metadata_generator import IMetadataGenerator
from app.utils.doc_parser import parse_file, RawFormat

logger = logging.getLogger(__name__)


class CleanupService:
    """
    Service for cleaning up and enriching documents with LLM-generated metadata.
    """
    
    def __init__(
        self,
        metadata_generator: IMetadataGenerator,
        source_base_dir: str = "data/company",
        output_base_dir: str = "cleaned/company"
    ):
        """
        Initialize the cleanup service.
        
        Args:
            metadata_generator: Metadata generator instance
            source_base_dir: Base directory for source documents
            output_base_dir: Base directory for cleaned/enriched documents
        """
        self.metadata_generator = metadata_generator
        self.source_base_dir = Path(source_base_dir)
        self.output_base_dir = Path(output_base_dir)
        
    async def cleanup_all_versions(self) -> CleanupReport:
        """
        Cleanup all version directories (v1, v2, etc.) in the source directory.
        
        Returns:
            CleanupReport with processing statistics
        """
        logger.info("Starting cleanup pipeline for all versions")
        
        report = CleanupReport()
        
        try:
            # Find all version directories (v1, v2, etc.)
            version_dirs = self._find_version_directories()
            
            if not version_dirs:
                logger.warning(f"No version directories found in {self.source_base_dir}")
                report.status = "completed"
                return report
            
            logger.info(f"Found {len(version_dirs)} version directories: {version_dirs}")
            
            # Process each version directory
            for version_dir in version_dirs:
                await self._process_version_directory(version_dir, report)
            
            # Mark report as completed
            report.mark_completed()
            
            logger.info(
                f"Cleanup completed: {report.successful_documents}/{report.total_documents} "
                f"documents processed successfully"
            )
            
            return report
            
        except Exception as e:
            logger.error(f"Cleanup pipeline failed: {e}", exc_info=True)
            report.status = "failed"
            report.errors.append(f"Pipeline error: {str(e)}")
            return report
    
    def _find_version_directories(self) -> List[str]:
        """
        Find all version directories (v1, v2, etc.) in the source directory.
        
        Returns:
            List of version directory names (e.g., ['v1', 'v2'])
        """
        if not self.source_base_dir.exists():
            logger.error(f"Source directory does not exist: {self.source_base_dir}")
            return []
        
        version_dirs = []
        for item in self.source_base_dir.iterdir():
            if item.is_dir() and item.name.startswith('v'):
                # Check if it's a valid version directory (v1, v2, etc.)
                version_num = item.name[1:]
                if version_num.isdigit():
                    version_dirs.append(item.name)
        
        return sorted(version_dirs)
    
    async def _process_version_directory(
        self, 
        version_dir: str, 
        report: CleanupReport
    ) -> None:
        """
        Process all documents in a version directory.
        
        Args:
            version_dir: Version directory name (e.g., 'v1')
            report: Cleanup report to update
        """
        source_dir = self.source_base_dir / version_dir
        output_dir = self.output_base_dir / version_dir
        
        logger.info(f"Processing version directory: {version_dir}")
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all document files (exclude .meta.json files)
        document_files = self._find_document_files(source_dir)
        
        logger.info(f"Found {len(document_files)} documents in {version_dir}")
        report.total_documents += len(document_files)
        
        # Process each document
        for doc_file in document_files:
            await self._process_document(
                source_dir=source_dir,
                output_dir=output_dir,
                document_file=doc_file,
                report=report
            )
    
    def _find_document_files(self, directory: Path) -> List[str]:
        """
        Find all document files in a directory (excluding .meta.json files).
        
        Args:
            directory: Directory to scan
            
        Returns:
            List of document filenames
        """
        if not directory.exists():
            return []
        
        document_files = []
        for item in directory.iterdir():
            if item.is_file() and not item.name.endswith('.meta.json'):
                document_files.append(item.name)
        
        return sorted(document_files)
    
    async def _process_document(
        self,
        source_dir: Path,
        output_dir: Path,
        document_file: str,
        report: CleanupReport
    ) -> None:
        """
        Process a single document: extract text, generate metadata, save enriched version.
        
        Args:
            source_dir: Source directory containing the document
            output_dir: Output directory for enriched document
            document_file: Document filename
            report: Cleanup report to update
        """
        start_time = time.time()
        document_id = Path(document_file).stem
        source_path = source_dir / document_file
        
        logger.info(f"Processing document: {document_id}")
        
        try:
            # 1. Extract text from document
            text = parse_file(str(source_path))
            
            if not text or len(text.strip()) < 10:
                logger.warning(f"Document {document_id} has insufficient content, skipping")
                status = DocumentCleanupStatus(
                    source_path=str(source_path),
                    document_id=document_id,
                    status="skipped",
                    error_message="Insufficient content"
                )
                report.add_document_status(status)
                return
            
            # 2. Load existing strict metadata
            strict_metadata = self._load_strict_metadata(source_dir, document_file)
            
            # 3. Generate soft metadata using LLM
            soft_metadata = await self.metadata_generator.generate_metadata(
                text=text,
                document_id=document_id,
                existing_metadata=strict_metadata.model_dump() if strict_metadata else None
            )
            
            # 4. Create enriched metadata
            processing_time = (time.time() - start_time) * 1000
            
            enriched_metadata = EnrichedMetadata(
                strict=strict_metadata,
                soft=soft_metadata,
                processing_time_ms=processing_time
            )
            
            # 5. Save enriched document and metadata
            output_path = self._save_enriched_document(
                output_dir=output_dir,
                document_file=document_file,
                text=text,
                enriched_metadata=enriched_metadata
            )
            
            # 6. Update report
            status = DocumentCleanupStatus(
                source_path=str(source_path),
                document_id=document_id,
                status="success",
                processing_time_ms=processing_time,
                enriched_path=str(output_path)
            )
            report.add_document_status(status)
            
            logger.info(
                f"Successfully processed {document_id} in {processing_time:.2f}ms"
            )
            
        except Exception as e:
            logger.error(f"Failed to process {document_id}: {e}", exc_info=True)
            
            processing_time = (time.time() - start_time) * 1000
            status = DocumentCleanupStatus(
                source_path=str(source_path),
                document_id=document_id,
                status="failed",
                error_message=str(e),
                processing_time_ms=processing_time
            )
            report.add_document_status(status)
    
    def _load_strict_metadata(
        self, 
        source_dir: Path, 
        document_file: str
    ) -> StrictMetadata:
        """
        Load strict metadata from existing .meta.json file.
        
        Args:
            source_dir: Directory containing the document
            document_file: Document filename
            
        Returns:
            StrictMetadata object
        """
        meta_file = source_dir / f"{Path(document_file).stem}.meta.json"
        
        if not meta_file.exists():
            logger.warning(f"No metadata file found for {document_file}, using defaults")
            return StrictMetadata(
                document_type="unknown",
                department="General",
                sensitivity="public",
                source=document_file
            )
        
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta_data = json.load(f)
            
            # Map existing metadata to StrictMetadata model
            return StrictMetadata(
                document_type=meta_data.get("document_type", "unknown"),
                department=meta_data.get("department", "General"),
                sensitivity=meta_data.get("sensitivity", "public"),
                source=document_file,
                domain=meta_data.get("domain"),
                published_year=meta_data.get("published_year"),
                effective_date=meta_data.get("effective_date", ""),
                allowed_roles=meta_data.get("allowed_roles"),
                tags=meta_data.get("tags", [])
            )
            
        except Exception as e:
            logger.error(f"Failed to load metadata from {meta_file}: {e}")
            return StrictMetadata(
                document_type="unknown",
                department="General",
                sensitivity="public",
                source=document_file
            )
    
    def _save_enriched_document(
        self,
        output_dir: Path,
        document_file: str,
        text: str,
        enriched_metadata: EnrichedMetadata
    ) -> Path:
        """
        Save enriched document and metadata to output directory.
        
        Args:
            output_dir: Output directory
            document_file: Original document filename
            text: Document text content
            enriched_metadata: Enriched metadata
            
        Returns:
            Path to saved document
        """
        # Save document text
        output_file = output_dir / document_file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)
        
        # Save enriched metadata
        meta_file = output_dir / f"{Path(document_file).stem}.enriched.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(enriched_metadata.model_dump(), f, indent=2, ensure_ascii=False)
        
        logger.debug(f"Saved enriched document to {output_file}")
        logger.debug(f"Saved enriched metadata to {meta_file}")
        
        return output_file
    
    async def preview_metadata(self, document_id: str, version: str = "v1") -> Optional[Dict[str, Any]]:
        """
        Preview original vs enriched metadata for a document.
        
        Args:
            document_id: Document identifier
            version: Version directory (e.g., 'v1')
            
        Returns:
            Dictionary with original and enriched metadata, or None if not found
        """
        source_dir = self.source_base_dir / version
        output_dir = self.output_base_dir / version
        
        # Find document file
        document_files = [
            f for f in self._find_document_files(source_dir)
            if Path(f).stem == document_id
        ]
        
        if not document_files:
            logger.warning(f"Document {document_id} not found in {version}")
            return None
        
        document_file = document_files[0]
        
        # Load original metadata
        original_meta = self._load_strict_metadata(source_dir, document_file)
        
        # Load enriched metadata if exists
        enriched_meta_file = output_dir / f"{document_id}.enriched.json"
        enriched_meta = None
        
        if enriched_meta_file.exists():
            try:
                with open(enriched_meta_file, 'r', encoding='utf-8') as f:
                    enriched_meta = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load enriched metadata: {e}")
        
        return {
            "document_id": document_id,
            "version": version,
            "source_path": str(source_dir / document_file),
            "original_metadata": original_meta.model_dump(),
            "enriched_metadata": enriched_meta,
            "has_enriched": enriched_meta is not None
        }
