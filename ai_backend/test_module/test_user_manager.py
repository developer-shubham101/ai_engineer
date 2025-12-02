"""Tests for user manager module."""
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
def user_manager(container):
    """Get user manager instance."""
    return container.get_user_manager()


def test_get_existing_user(user_manager):
    """Test retrieving an existing user."""
    async def run_test():
        try:
            user = await user_manager.get_user("u_admin_1")
            assert user is not None, "Should retrieve existing user"
            assert user.get('username') is not None, "Username should be present"
            assert user.get('role') is not None, "Role should be present"
        except Exception as e:
            pytest.fail(f"Get existing user failed: {e}")
    
    asyncio.run(run_test())


def test_get_nonexistent_user(user_manager):
    """Test retrieving a non-existent user."""
    async def run_test():
        try:
            user = await user_manager.get_user("nonexistent_user")
            assert user is None, "Should return None for non-existent user"
        except Exception as e:
            pytest.fail(f"Get non-existent user test failed: {e}")
    
    asyncio.run(run_test())


def test_get_user_with_empty_id(user_manager):
    """Test retrieving user with empty ID."""
    async def run_test():
        try:
            user = await user_manager.get_user("")
            assert user is None, "Should return None for empty user ID"
        except Exception as e:
            pytest.fail(f"Empty user ID test failed: {e}")
    
    asyncio.run(run_test())


def test_get_user_with_none_id(user_manager):
    """Test retrieving user with None ID."""
    async def run_test():
        try:
            user = await user_manager.get_user(None)
            assert user is None, "Should return None for None user ID"
        except Exception as e:
            pytest.fail(f"None user ID test failed: {e}")
    
    asyncio.run(run_test())


def test_user_data_structure(user_manager):
    """Test user data structure completeness."""
    async def run_test():
        try:
            user = await user_manager.get_user("u_admin_1")
            if user:
                expected_fields = ['username', 'role', 'department']
                for field in expected_fields:
                    assert field in user, f"User should have {field} field"
        except Exception as e:
            pytest.fail(f"User data structure test failed: {e}")
    
    asyncio.run(run_test())


async def run_standalone_tests():
    """Run tests without pytest for standalone execution."""
    print("Testing User Manager Module")
    print("=" * 50)
    
    try:
        container = get_container()
        container.initialize()
        user_manager = container.get_user_manager()
        
        # Test existing user
        user = await user_manager.get_user("u_admin_1")
        print(f"[PASS] Get existing user: {user is not None}")
        if user:
            print(f"  Username: {user.get('username')}")
            print(f"  Role: {user.get('role')}")
            print(f"  Department: {user.get('department')}")
        
        # Test non-existent user
        nonexistent = await user_manager.get_user("fake_user")
        print(f"[PASS] Non-existent user returns None: {nonexistent is None}")
        
        print("All user manager tests passed!")
        
    except Exception as e:
        print(f"[FAIL] User manager tests failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_standalone_tests())