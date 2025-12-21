"""Tests for vector store module."""
import asyncio
import sys
from pathlib import Path
import pytest

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.modules.integration import get_container


@pytest.fixture
def container():
    """Setup container for tests."""
    container = get_container()
    container.initialize()
    return container


@pytest.fixture
def vector_store(container):
    """Get vector store instance."""
    return container.get_vector_store()


def test_vector_store_initialization(vector_store):
    """Test vector store initialization."""
    try:
        assert vector_store is not None, "Vector store should be initialized"
        assert hasattr(vector_store, 'collection_name'), "Should have collection_name attribute"
    except Exception as e:
        pytest.fail(f"Vector store initialization failed: {e}")


def test_collection_name(vector_store):
    """Test collection name property."""
    try:
        collection_name = vector_store.collection_name
        assert collection_name is not None, "Collection name should not be None"
        assert isinstance(collection_name, str), "Collection name should be string"
    except Exception as e:
        pytest.fail(f"Collection name test failed: {e}")


def test_vector_store_accessibility(vector_store):
    """Test if vector store is accessible."""
    try:
        # Try to access basic properties without querying
        assert vector_store is not None, "Vector store should be accessible"
        # Additional checks can be added based on vector store interface
    except Exception as e:
        pytest.fail(f"Vector store accessibility test failed: {e}")


def test_vector_store_methods(vector_store):
    """Test vector store has required methods."""
    try:
        required_methods = ['collection_name']  # Add more as needed
        for method in required_methods:
            assert hasattr(vector_store, method), f"Should have {method} method/property"
    except Exception as e:
        pytest.fail(f"Vector store methods test failed: {e}")


async def run_standalone_tests():
    """Run tests without pytest for standalone execution."""
    print("Testing Vector Store Module")
    print("=" * 50)
    
    try:
        container = get_container()
        container.initialize()
        vector_store = container.get_vector_store()
        
        print("[PASS] Vector store initialized successfully")
        if hasattr(vector_store, 'collection_name'):
            print(f"  Collection: {vector_store.collection_name}")
        
        print("All vector store tests passed!")
        
    except Exception as e:
        print(f"[FAIL] Vector store tests failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_standalone_tests())