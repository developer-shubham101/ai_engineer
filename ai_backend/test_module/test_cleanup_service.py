"""
Integration tests for cleanup service.
"""
import pytest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, AsyncMock

from app.modules.core.cleanup_service import CleanupService
from app.modules.core.metadata_models import SoftMetadata, CleanupReport


@pytest.fixture
def temp_directories():
    """Create temporary source and output directories."""
    temp_dir = tempfile.mkdtemp()
    source_dir = Path(temp_dir) / "source"
    output_dir = Path(temp_dir) / "output"
    
    # Create source directory structure
    source_dir.mkdir(parents=True)
    (source_dir / "v1").mkdir()
    (source_dir / "v2").mkdir()
    
    yield source_dir, output_dir
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_metadata_generator():
    """Create a mock metadata generator."""
    generator = Mock()
    
    # Mock generate_metadata to return sample soft metadata
    async def mock_generate(text, document_id, existing_metadata=None):
        return SoftMetadata(
            summary=f"Summary of {document_id}",
            keywords=["test", "document", "sample"],
            themes=["testing", "documentation"],
            entities={"people": ["Test User"], "organizations": [], "locations": []},
            llm_model="test-model",
            confidence=0.9
        )
    
    generator.generate_metadata = AsyncMock(side_effect=mock_generate)
    return generator


@pytest.fixture
def cleanup_service(temp_directories, mock_metadata_generator):
    """Create cleanup service with temp directories."""
    source_dir, output_dir = temp_directories
    return CleanupService(
        metadata_generator=mock_metadata_generator,
        source_base_dir=str(source_dir),
        output_base_dir=str(output_dir)
    )


def create_test_document(directory: Path, filename: str, content: str, metadata: dict):
    """Helper to create a test document with metadata."""
    # Create document file
    doc_path = directory / filename
    with open(doc_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    # Create metadata file
    meta_path = directory / f"{Path(filename).stem}.meta.json"
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f)


@pytest.mark.asyncio
async def test_cleanup_all_versions_success(cleanup_service, temp_directories):
    """Test successful cleanup of all version directories."""
    source_dir, output_dir = temp_directories
    
    # Create test documents in v1
    create_test_document(
        source_dir / "v1",
        "test_doc_1.md",
        "# Test Document 1\n\nThis is a test document.",
        {"document_type": "memo", "department": "HR", "sensitivity": "public", "tags": []}
    )
    
    create_test_document(
        source_dir / "v1",
        "test_doc_2.md",
        "# Test Document 2\n\nAnother test document.",
        {"document_type": "policy", "department": "IT", "sensitivity": "confidential", "tags": []}
    )
    
    # Create test document in v2
    create_test_document(
        source_dir / "v2",
        "test_doc_3.md",
        "# Test Document 3\n\nYet another test document.",
        {"document_type": "report", "department": "Finance", "sensitivity": "public", "tags": []}
    )
    
    # Run cleanup
    report = await cleanup_service.cleanup_all_versions()
    
    # Assertions
    assert isinstance(report, CleanupReport)
    assert report.status == "completed"
    assert report.total_documents == 3
    assert report.successful_documents == 3
    assert report.failed_documents == 0
    
    # Verify output files exist
    assert (output_dir / "v1" / "test_doc_1.md").exists()
    assert (output_dir / "v1" / "test_doc_1.enriched.json").exists()
    assert (output_dir / "v1" / "test_doc_2.md").exists()
    assert (output_dir / "v1" / "test_doc_2.enriched.json").exists()
    assert (output_dir / "v2" / "test_doc_3.md").exists()
    assert (output_dir / "v2" / "test_doc_3.enriched.json").exists()
    
    # Verify enriched metadata structure
    with open(output_dir / "v1" / "test_doc_1.enriched.json", 'r') as f:
        enriched = json.load(f)
        assert "strict" in enriched
        assert "soft" in enriched
        assert enriched["strict"]["document_type"] == "memo"
        assert enriched["soft"]["summary"] == "Summary of test_doc_1"


@pytest.mark.asyncio
async def test_cleanup_empty_directory(cleanup_service, temp_directories):
    """Test cleanup with no documents."""
    # Run cleanup on empty directories
    report = await cleanup_service.cleanup_all_versions()
    
    # Assertions
    assert isinstance(report, CleanupReport)
    assert report.status == "completed"
    assert report.total_documents == 0
    assert report.successful_documents == 0


@pytest.mark.asyncio
async def test_cleanup_with_insufficient_content(cleanup_service, temp_directories):
    """Test cleanup skips documents with insufficient content."""
    source_dir, output_dir = temp_directories
    
    # Create document with very little content
    create_test_document(
        source_dir / "v1",
        "empty_doc.md",
        "x",  # Only 1 character
        {"document_type": "memo", "department": "HR", "sensitivity": "public", "tags": []}
    )
    
    # Run cleanup
    report = await cleanup_service.cleanup_all_versions()
    
    # Assertions
    assert report.total_documents == 1
    assert report.skipped_documents == 1
    assert report.successful_documents == 0


@pytest.mark.asyncio
async def test_cleanup_with_metadata_generation_failure(cleanup_service, temp_directories, mock_metadata_generator):
    """Test cleanup handles metadata generation failures."""
    source_dir, output_dir = temp_directories
    
    # Make metadata generator fail
    mock_metadata_generator.generate_metadata = AsyncMock(
        side_effect=Exception("Metadata generation failed")
    )
    
    # Create test document
    create_test_document(
        source_dir / "v1",
        "test_doc.md",
        "# Test Document\n\nThis is a test.",
        {"document_type": "memo", "department": "HR", "sensitivity": "public", "tags": []}
    )
    
    # Run cleanup
    report = await cleanup_service.cleanup_all_versions()
    
    # Assertions
    assert report.total_documents == 1
    assert report.failed_documents == 1
    assert report.successful_documents == 0
    assert len(report.errors) > 0


@pytest.mark.asyncio
async def test_find_version_directories(cleanup_service, temp_directories):
    """Test finding version directories."""
    source_dir, _ = temp_directories
    
    # Create additional version directories
    (source_dir / "v3").mkdir()
    (source_dir / "v10").mkdir()
    (source_dir / "not_a_version").mkdir()  # Should be ignored
    
    # Find version directories
    versions = cleanup_service._find_version_directories()
    
    # Assertions
    assert "v1" in versions
    assert "v2" in versions
    assert "v3" in versions
    assert "v10" in versions
    assert "not_a_version" not in versions
    assert versions == sorted(versions)  # Should be sorted


@pytest.mark.asyncio
async def test_preview_metadata(cleanup_service, temp_directories):
    """Test metadata preview functionality."""
    source_dir, output_dir = temp_directories
    
    # Create test document
    create_test_document(
        source_dir / "v1",
        "preview_doc.md",
        "# Preview Document\n\nThis is for preview.",
        {"document_type": "memo", "department": "HR", "sensitivity": "public", "tags": ["test"]}
    )
    
    # First preview - before enrichment
    preview = await cleanup_service.preview_metadata("preview_doc", "v1")
    
    assert preview is not None
    assert preview["document_id"] == "preview_doc"
    assert preview["version"] == "v1"
    assert preview["has_enriched"] is False
    assert preview["original_metadata"]["document_type"] == "memo"
    
    # Run cleanup to create enriched version
    await cleanup_service.cleanup_all_versions()
    
    # Second preview - after enrichment
    preview = await cleanup_service.preview_metadata("preview_doc", "v1")
    
    assert preview["has_enriched"] is True
    assert preview["enriched_metadata"] is not None
    assert "strict" in preview["enriched_metadata"]
    assert "soft" in preview["enriched_metadata"]


@pytest.mark.asyncio
async def test_preview_metadata_not_found(cleanup_service):
    """Test preview for non-existent document."""
    preview = await cleanup_service.preview_metadata("nonexistent_doc", "v1")
    assert preview is None


@pytest.mark.asyncio
async def test_cleanup_preserves_original_files(cleanup_service, temp_directories):
    """Test that cleanup doesn't modify original files."""
    source_dir, output_dir = temp_directories
    
    original_content = "# Original Document\n\nOriginal content."
    
    # Create test document
    create_test_document(
        source_dir / "v1",
        "preserve_test.md",
        original_content,
        {"document_type": "memo", "department": "HR", "sensitivity": "public", "tags": []}
    )
    
    # Run cleanup
    await cleanup_service.cleanup_all_versions()
    
    # Verify original file is unchanged
    with open(source_dir / "v1" / "preserve_test.md", 'r') as f:
        content = f.read()
        assert content == original_content
    
    # Verify original metadata is unchanged
    with open(source_dir / "v1" / "preserve_test.meta.json", 'r') as f:
        metadata = json.load(f)
        assert metadata["document_type"] == "memo"


@pytest.mark.asyncio
async def test_cleanup_report_statistics(cleanup_service, temp_directories):
    """Test cleanup report calculates statistics correctly."""
    source_dir, _ = temp_directories
    
    # Create multiple test documents
    for i in range(5):
        create_test_document(
            source_dir / "v1",
            f"doc_{i}.md",
            f"# Document {i}\n\nContent for document {i}.",
            {"document_type": "memo", "department": "HR", "sensitivity": "public", "tags": []}
        )
    
    # Run cleanup
    report = await cleanup_service.cleanup_all_versions()
    
    # Assertions
    assert report.total_documents == 5
    assert report.processed_documents == 5
    assert report.successful_documents == 5
    assert report.average_processing_time_ms is not None
    assert report.average_processing_time_ms > 0
    assert report.total_processing_time_ms is not None
    assert len(report.document_statuses) == 5
