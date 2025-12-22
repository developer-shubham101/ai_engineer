"""Tests for FaissVectorStore."""
import asyncio
import os
import sys
from pathlib import Path
import pytest
import numpy as np

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Set environment variable to use Faiss
os.environ["VECTOR_STORE_TYPE"] = "faiss"

from app.modules.integration import get_container, reset_container



@pytest.fixture(scope="module")
def faiss_container():
    """Setup container for tests."""
    # Reset container to ensure clean state
    reset_container()
    container = get_container()
    container.initialize()
    # Override file path for testing
    container.get_vector_store().file_path = "vector_db/faiss_db/test_faiss_index.pkl"
    # Force reload to pick up the new path
    container.get_vector_store()._load_index()
    yield container
    # Cleanup after tests
    if os.path.exists("vector_db/faiss_db/test_faiss_index.pkl"):
        try:
            os.remove("vector_db/faiss_db/test_faiss_index.pkl")
        except PermissionError:
            pass

@pytest.fixture(scope="module")
def vector_store(faiss_container):
    """Get vector store instance."""
    return faiss_container.get_vector_store()

@pytest.mark.asyncio
async def test_vector_store_initialization(vector_store):
    """Test vector store initialization."""
    assert vector_store is not None, "Vector store should be initialized"
    assert "faiss" in vector_store.__class__.__name__.lower(), "Should be FaissVectorStore"
    assert vector_store.index is not None, "FAISS index should be initialized"

@pytest.mark.asyncio
async def test_add_and_search_document(vector_store):
    """Test adding and searching a document."""
    text = "This is a test document about FAISS."
    metadata = {"source": "test"}
    doc_id = await vector_store.add_document(text, metadata)
    
    # Ensure document was added
    assert vector_store.index.ntotal == 1
    
    # Search for the document
    results = await vector_store.search_documents("test FAISS", top_k=1)
    
    assert len(results) == 1
    assert results[0]["text"] == text
    assert results[0]["metadata"]["source"] == "test"

@pytest.mark.asyncio
async def test_persistence(faiss_container):
    """Test if the index is saved and loaded correctly."""
    # Add a document to the first store
    store1 = faiss_container.get_vector_store()
    await store1.add_document("Document for persistence test.", {"source": "persistence"})

    # Create a new container and vector store, which should load from the same file
    reset_container()
    new_container = get_container()
    new_container.initialize()
    new_container.get_vector_store().file_path = "vector_db/faiss_db/test_faiss_index.pkl"
    store2 = new_container.get_vector_store()
    store2._load_index()
    
    # The new store should have the document from the first store
    assert store2.index.ntotal > 0
    results = await store2.search_documents("persistence test", top_k=1)
    assert len(results) > 0
    assert "Document for persistence test." in [r['text'] for r in results]

if __name__ == "__main__":
    pytest.main([__file__])
