"""
Unit tests for metadata generator.
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch

from app.modules.core.metadata_generator import LLMMetadataGenerator
from app.modules.core.metadata_models import SoftMetadata
from app.modules.llm.interfaces import LLMResponse


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    provider = Mock()
    provider.get_model_name.return_value = "test-model"
    return provider


@pytest.fixture
def metadata_generator(mock_llm_provider):
    """Create a metadata generator with mock LLM provider."""
    return LLMMetadataGenerator(mock_llm_provider)


@pytest.mark.asyncio
async def test_generate_metadata_success(metadata_generator, mock_llm_provider):
    """Test successful metadata generation."""
    # Mock LLM response
    llm_response = LLMResponse(
        text="""SUMMARY: This is a test document about company policies.
KEYWORDS: policy, test, company, document, guidelines
THEMES: corporate policy, documentation, guidelines
PEOPLE: John Doe, Jane Smith
ORGANIZATIONS: Acme Corp, Test Inc
LOCATIONS: New York, San Francisco""",
        metadata={},
        usage={},
        finish_reason="completed"
    )
    mock_llm_provider.generate = AsyncMock(return_value=llm_response)
    
    # Test document
    text = "This is a test document about company policies."
    document_id = "test_doc_001"
    
    # Generate metadata
    result = await metadata_generator.generate_metadata(text, document_id)
    
    # Assertions
    assert isinstance(result, SoftMetadata)
    assert "test document" in result.summary.lower()
    assert "policy" in result.keywords
    assert len(result.keywords) > 0
    assert len(result.themes) > 0
    assert "people" in result.entities
    assert "John Doe" in result.entities["people"]
    assert result.llm_model == "test-model"


@pytest.mark.asyncio
async def test_generate_metadata_with_truncation(metadata_generator, mock_llm_provider):
    """Test metadata generation with text truncation."""
    # Mock LLM response
    llm_response = LLMResponse(
        text="""SUMMARY: Long document summary.
KEYWORDS: test, long, document
THEMES: testing
PEOPLE: None
ORGANIZATIONS: None
LOCATIONS: None""",
        metadata={},
        usage={},
        finish_reason="completed"
    )
    mock_llm_provider.generate = AsyncMock(return_value=llm_response)
    
    # Create very long text
    long_text = "Test content. " * 1000  # Much longer than max_text_length
    document_id = "long_doc_001"
    
    # Generate metadata
    result = await metadata_generator.generate_metadata(long_text, document_id)
    
    # Assertions
    assert isinstance(result, SoftMetadata)
    assert len(result.keywords) > 0
    
    # Verify LLM was called with truncated text
    call_args = mock_llm_provider.generate.call_args
    prompt = call_args.kwargs['prompt']
    assert "[... content truncated ...]" in prompt or len(prompt) < len(long_text)


@pytest.mark.asyncio
async def test_generate_metadata_llm_failure(metadata_generator, mock_llm_provider):
    """Test metadata generation when LLM fails."""
    # Mock LLM to raise exception
    mock_llm_provider.generate = AsyncMock(side_effect=Exception("LLM error"))
    
    text = "Test document"
    document_id = "test_doc_002"
    
    # Generate metadata (should return fallback)
    result = await metadata_generator.generate_metadata(text, document_id)
    
    # Assertions - should return fallback metadata
    assert isinstance(result, SoftMetadata)
    assert "fallback" in result.summary.lower()
    assert len(result.keywords) > 0
    assert result.confidence == 0.3  # Low confidence for fallback


@pytest.mark.asyncio
async def test_generate_metadata_malformed_response(metadata_generator, mock_llm_provider):
    """Test metadata generation with malformed LLM response."""
    # Mock LLM response with malformed data
    llm_response = LLMResponse(
        text="This is not a properly formatted response",
        metadata={},
        usage={},
        finish_reason="completed"
    )
    mock_llm_provider.generate = AsyncMock(return_value=llm_response)
    
    text = "Test document"
    document_id = "test_doc_003"
    
    # Generate metadata
    result = await metadata_generator.generate_metadata(text, document_id)
    
    # Assertions - should handle gracefully with defaults
    assert isinstance(result, SoftMetadata)
    assert len(result.summary) > 0
    assert len(result.keywords) > 0
    assert len(result.themes) > 0


@pytest.mark.asyncio
async def test_generate_metadata_with_existing_metadata(metadata_generator, mock_llm_provider):
    """Test metadata generation with existing metadata context."""
    # Mock LLM response
    llm_response = LLMResponse(
        text="""SUMMARY: Document summary.
KEYWORDS: test, metadata
THEMES: testing
PEOPLE: None
ORGANIZATIONS: None
LOCATIONS: None""",
        metadata={},
        usage={},
        finish_reason="completed"
    )
    mock_llm_provider.generate = AsyncMock(return_value=llm_response)
    
    text = "Test document"
    document_id = "test_doc_004"
    existing_metadata = {
        "document_type": "memo",
        "department": "HR"
    }
    
    # Generate metadata
    result = await metadata_generator.generate_metadata(
        text, 
        document_id, 
        existing_metadata
    )
    
    # Assertions
    assert isinstance(result, SoftMetadata)
    # Existing metadata is passed but doesn't affect soft metadata structure
    assert len(result.keywords) > 0


def test_truncate_text(metadata_generator):
    """Test text truncation logic."""
    # Short text - should not be truncated
    short_text = "Short text"
    result = metadata_generator._truncate_text(short_text)
    assert result == short_text
    
    # Long text - should be truncated
    long_text = "Test " * 2000
    result = metadata_generator._truncate_text(long_text)
    assert len(result) < len(long_text)
    assert "[... content truncated ...]" in result


def test_parse_llm_response(metadata_generator):
    """Test LLM response parsing."""
    response_text = """SUMMARY: Test summary here.
KEYWORDS: key1, key2, key3
THEMES: theme1, theme2
PEOPLE: Person A, Person B
ORGANIZATIONS: Org A
LOCATIONS: Location A, Location B"""
    
    result = metadata_generator._parse_llm_response(response_text, "test_doc")
    
    assert isinstance(result, SoftMetadata)
    assert result.summary == "Test summary here."
    assert "key1" in result.keywords
    assert "theme1" in result.themes
    assert "Person A" in result.entities["people"]
    assert "Org A" in result.entities["organizations"]
    assert "Location A" in result.entities["locations"]


def test_create_fallback_metadata(metadata_generator):
    """Test fallback metadata creation."""
    text = "This is a test document with some longer words like documentation and implementation"
    document_id = "fallback_doc"
    
    result = metadata_generator._create_fallback_metadata(text, document_id)
    
    assert isinstance(result, SoftMetadata)
    assert "fallback" in result.summary.lower()
    assert len(result.keywords) > 0
    assert result.confidence == 0.3
    assert result.themes == ["general"]
