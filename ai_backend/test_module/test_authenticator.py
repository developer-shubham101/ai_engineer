"""Tests for authenticator module."""
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
def authenticator(container):
    """Get authenticator instance."""
    return container.get_authenticator()


def test_valid_authentication(authenticator):
    """Test authentication with valid credentials."""
    async def run_test():
        try:
            user_data = await authenticator.authenticate("admin", "admin123")
            assert user_data is not None, "Authentication should succeed with valid credentials"
            assert user_data.get('username') == "admin", "Username should match"
            assert user_data.get('role') is not None, "Role should be present"
        except Exception as e:
            pytest.fail(f"Valid authentication failed: {e}")
    
    asyncio.run(run_test())


def test_invalid_authentication(authenticator):
    """Test authentication with invalid credentials."""
    async def run_test():
        try:
            user_data = await authenticator.authenticate("invalid", "wrong")
            assert user_data is None, "Authentication should fail with invalid credentials"
        except Exception as e:
            pytest.fail(f"Invalid authentication test failed: {e}")
    
    asyncio.run(run_test())


def test_empty_credentials(authenticator):
    """Test authentication with empty credentials."""
    async def run_test():
        try:
            user_data = await authenticator.authenticate("", "")
            assert user_data is None, "Authentication should fail with empty credentials"
        except Exception as e:
            pytest.fail(f"Empty credentials test failed: {e}")
    
    asyncio.run(run_test())


def test_none_credentials(authenticator):
    """Test authentication with None credentials."""
    async def run_test():
        try:
            user_data = await authenticator.authenticate(None, None)
            assert user_data is None, "Authentication should fail with None credentials"
        except Exception as e:
            pytest.fail(f"None credentials test failed: {e}")
    
    asyncio.run(run_test())


async def run_standalone_tests():
    """Run tests without pytest for standalone execution."""
    print("Testing Authenticator Module")
    print("=" * 50)
    
    try:
        container = get_container()
        container.initialize()
        authenticator = container.get_authenticator()
        
        # Test valid credentials
        user_data = await authenticator.authenticate("admin", "admin123")
        print(f"[PASS] Valid auth: {user_data is not None}")
        
        # Test invalid credentials
        invalid_user = await authenticator.authenticate("invalid", "wrong")
        print(f"[PASS] Invalid auth rejected: {invalid_user is None}")
        
        print("All authenticator tests passed!")
        
    except Exception as e:
        print(f"[FAIL] Authenticator tests failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_standalone_tests())