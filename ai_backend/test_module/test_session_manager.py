"""Tests for session manager module."""
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
def session_manager(container):
    """Get session manager instance."""
    return container.get_session_manager()


def test_create_session(session_manager):
    """Test session creation."""
    try:
        session_id = session_manager.create_session(
            session_id=None,
            role="Employee",
            department="Engineering"
        )
        assert session_id is not None, "Session ID should be generated"
        assert isinstance(session_id, str), "Session ID should be string"
    except Exception as e:
        pytest.fail(f"Session creation failed: {e}")


def test_create_session_with_custom_id(session_manager):
    """Test session creation with custom ID."""
    import uuid
    try:
        custom_id = f"test_session_{uuid.uuid4().hex[:8]}"
        session_id = session_manager.create_session(
            session_id=custom_id,
            role="Manager",
            department="HR"
        )
        assert session_id == custom_id, "Should use provided session ID"
    except Exception as e:
        pytest.fail(f"Custom session ID creation failed: {e}")


def test_store_message(session_manager):
    """Test message storage."""
    try:
        session_id = session_manager.create_session(
            session_id=None,
            role="Employee",
            department="Engineering"
        )
        
        msg_id = session_manager.store_message(
            session_id=session_id,
            speaker="user",
            content="Test message"
        )
        assert msg_id is not None, "Message ID should be generated"
    except Exception as e:
        pytest.fail(f"Message storage failed: {e}")


def test_fetch_recent_messages(session_manager):
    """Test fetching recent messages."""
    try:
        session_id = session_manager.create_session(
            session_id=None,
            role="Employee",
            department="Engineering"
        )
        
        # Store multiple messages
        for i in range(3):
            session_manager.store_message(
                session_id=session_id,
                speaker="user",
                content=f"Test message {i}"
            )
        
        messages = session_manager.fetch_recent_messages(session_id, limit=5)
        assert isinstance(messages, list), "Should return list of messages"
        assert len(messages) <= 5, "Should respect limit parameter"
    except Exception as e:
        pytest.fail(f"Fetch recent messages failed: {e}")


def test_fetch_messages_empty_session(session_manager):
    """Test fetching messages from empty session."""
    try:
        session_id = session_manager.create_session(
            session_id=None,
            role="Employee",
            department="Engineering"
        )
        
        messages = session_manager.fetch_recent_messages(session_id, limit=5)
        assert isinstance(messages, list), "Should return empty list"
        assert len(messages) == 0, "Should be empty for new session"
    except Exception as e:
        pytest.fail(f"Empty session messages test failed: {e}")


def test_fetch_messages_nonexistent_session(session_manager):
    """Test fetching messages from non-existent session."""
    try:
        messages = session_manager.fetch_recent_messages("fake_session", limit=5)
        assert isinstance(messages, list), "Should return empty list"
        assert len(messages) == 0, "Should be empty for non-existent session"
    except Exception as e:
        pytest.fail(f"Non-existent session test failed: {e}")


async def run_standalone_tests():
    """Run tests without pytest for standalone execution."""
    print("Testing Session Manager Module")
    print("=" * 50)
    
    try:
        container = get_container()
        container.initialize()
        session_manager = container.get_session_manager()
        
        # Test session creation
        session_id = session_manager.create_session(
            session_id=None,
            role="Employee",
            department="Engineering"
        )
        print(f"[PASS] Session created: {session_id}")
        
        # Test message storage
        msg_id = session_manager.store_message(
            session_id=session_id,
            speaker="user",
            content="Hello, this is a test message"
        )
        print(f"[PASS] Message stored: {msg_id}")
        
        # Test message retrieval
        messages = session_manager.fetch_recent_messages(session_id, limit=5)
        print(f"[PASS] Retrieved {len(messages)} messages")
        
        print("All session manager tests passed!")
        
    except Exception as e:
        print(f"[FAIL] Session manager tests failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(run_standalone_tests())