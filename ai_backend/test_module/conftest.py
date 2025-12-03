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