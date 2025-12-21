"""Pytest configuration and shared fixtures."""
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.modules.integration import get_container


@pytest.fixture(scope="session")
def shared_container():
    """Shared container instance for all tests."""
    container = get_container()
    container.initialize()
    return container


@pytest.fixture
def fresh_container():
    """Fresh container instance for each test."""
    container = get_container()
    container.initialize()
    return container


# Exclude standalone scripts from pytest collection
collect_ignore = [
    "test_runner.py", 
    "test_rbac_verification.py", 
    "test_versioning_flow.py", 
    "test_api_metadata_validation.py", 
    "test_live_endpoints.py",
    "test_optimized_prompt.py",
    "test_conversation_context.py",
    "test_temperature.py",
    "test_simple_prompt.py"
]