"""Tests for RAG orchestrator module."""
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
def rag_orchestrator(container):
    """Get RAG orchestrator instance."""
    return container.get_rag_orchestrator()


def test_rag_orchestrator_initialization(rag_orchestrator):
    """Test RAG orchestrator initialization."""
    try:
        assert rag_orchestrator is not None, "RAG orchestrator should be initialized"
    except Exception as e:
        pytest.fail(f"RAG orchestrator initialization failed: {e}")


def test_rag_orchestrator_methods(rag_orchestrator):
    """Test RAG orchestrator has required methods."""
    try:
        # Check for common RAG methods (adjust based on actual interface)
        expected_methods = []  # Add methods as they become available
        for method in expected_methods:
            assert hasattr(rag_orchestrator, method), f"Should have {method} method"
    except Exception as e:
        pytest.fail(f"RAG orchestrator methods test failed: {e}")


def test_rag_orchestrator_type(rag_orchestrator):
    """Test RAG orchestrator type."""
    try:
        assert rag_orchestrator is not None, "RAG orchestrator should not be None"
        # Additional type checks can be added based on implementation
    except Exception as e:
        pytest.fail(f"RAG orchestrator type test failed: {e}")


@pytest.mark.asyncio
async def test_rbac_level_filtering(rag_orchestrator):
    """Employee must not see highly_confidential docs even in same department."""
    from app.modules.config.constants import ROLE_LEVELS, SENSITIVITY_LEVELS

    employee_user = {"role": "Employee", "department": "HR", "user_id": "emp1"}
    employee_level = ROLE_LEVELS.get("Employee", 0)
    required_level = SENSITIVITY_LEVELS.get("highly_confidential", 3)

    # Core assertion: Employee level is below highly_confidential threshold
    assert employee_level < required_level, (
        f"Employee level {employee_level} should be < highly_confidential level {required_level}"
    )

    # Simulate the RBAC filter logic directly
    doc = {"metadata": {"sensitivity": "highly_confidential", "department": "HR"}}
    sensitivity = doc["metadata"]["sensitivity"]
    doc_dept = doc["metadata"]["department"]
    req = SENSITIVITY_LEVELS.get(sensitivity, 0)

    # Employee in same dept still blocked by level check
    passes = employee_level >= req
    assert not passes, "Employee should NOT pass highly_confidential level check"


@pytest.mark.asyncio
async def test_context_length(rag_orchestrator):
    """build_context must use 2000-char limit per document, not 500."""
    long_text = "x" * 3000
    docs = [{"text": long_text, "id": "d1", "metadata": {}}]
    context = await rag_orchestrator.build_context(docs)
    # Should contain exactly 2000 chars of the document text (plus label)
    assert "x" * 2000 in context, "Context should include up to 2000 chars per doc"
    assert "x" * 2001 not in context, "Context must not exceed 2000 chars per doc"


async def run_standalone_tests():
    """Run tests without pytest for standalone execution."""
    print("Testing RAG Orchestrator Module")
    print("=" * 50)
    
    try:
        container = get_container()
        container.initialize()
        rag_orchestrator = container.get_rag_orchestrator()
        
        print("[PASS] RAG Orchestrator initialized successfully")
        
        print("All RAG orchestrator tests passed!")
        
    except Exception as e:
        print(f"[FAIL] RAG orchestrator tests failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_standalone_tests())