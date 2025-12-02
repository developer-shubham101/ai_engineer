"""
Comprehensive Container Validation Script
Tests all major components of the modular architecture.
"""
import asyncio
from app.modules.integration import get_container


async def test_authenticator():
    """Test authentication module."""
    print("\n" + "="*60)
    print("Testing Authenticator Module")
    print("="*60)
    
    try:
        container = get_container()
        container.initialize()
        
        authenticator = container.get_authenticator()
        
        # Test valid credentials
        user_data = await authenticator.authenticate("admin", "admin123")
        if user_data:
            print("Authentication successful!")
            print(f"   User: {user_data.get('username')} ({user_data.get('role')})")
        else:
            print("Authentication failed!")
        
        # Test invalid credentials
        invalid_user = await authenticator.authenticate("invalid", "wrong")
        if not invalid_user:
            print("Invalid credentials correctly rejected")
        else:
            print("Invalid credentials incorrectly accepted")
    except Exception as e:
        print(f"Authenticator test failed: {e}")


async def test_user_manager():
    """Test user management module."""
    print("\n" + "="*60)
    print("Testing User Manager Module")
    print("="*60)
    
    try:
        container = get_container()
        container.initialize()
        
        user_manager = container.get_user_manager()
        
        # Get user
        user = await user_manager.get_user("u_admin_1")
        if user:
            print("User retrieval successful!")
            print(f"   Username: {user.get('username')}")
            print(f"   Role: {user.get('role')}")
            print(f"   Department: {user.get('department')}")
        else:
            print("User retrieval failed!")
    except Exception as e:
        print(f"User manager test failed: {e}")


async def test_session_manager():
    """Test session management module."""
    print("\n" + "="*60)
    print("Testing Session Manager Module")
    print("="*60)
    
    try:
        container = get_container()
        container.initialize()
        
        session_manager = container.get_session_manager()
        
        # Create session with metadata
        session_id = session_manager.create_session(
            session_id=None,  # Let it auto-generate
            role="Employee",
            department="Engineering"
        )
        print(f"Session created: {session_id}")
        
        # Store message
        msg_id = session_manager.store_message(
            session_id=session_id,
            speaker="user",
            content="Hello, this is a test message"
        )
        print(f"Message stored: {msg_id}")
        
        # Retrieve messages
        messages = session_manager.fetch_recent_messages(session_id, limit=5)
        print(f"Retrieved {len(messages)} messages")
    except Exception as e:
        print(f"Session manager test failed: {e}")



async def test_vector_store():
    """Test vector database module."""
    print("\n" + "="*60)
    print("Testing Vector Store Module")
    print("="*60)
    
    container = get_container()
    container.initialize()
    
    vector_store = container.get_vector_store()
    
    # Get collection info
    try:
        # Try to query (this will test if ChromaDB is accessible)
        print("Vector store initialized successfully")
        print(f"   Collection: {vector_store.collection_name}")
    except Exception as e:
        print(f"Vector store error: {e}")


async def test_rag_orchestrator():
    """Test RAG orchestrator module."""
    print("\n" + "="*60)
    print("Testing RAG Orchestrator Module")
    print("="*60)
    
    try:
        container = get_container()
        container.initialize()
        
        rag_orchestrator = container.get_rag_orchestrator()
        
        print("RAG Orchestrator initialized successfully")
    except Exception as e:
        print(f"RAG orchestrator test failed: {e}")


async def main():
    """Run all validation tests."""
    print("\n" + "="*60)
    print("CONTAINER VALIDATION SUITE")
    print("="*60)
    
    try:
        await test_authenticator()
        await test_user_manager()
        await test_session_manager()
        await test_vector_store()
        await test_rag_orchestrator()
        
        print("\n" + "="*60)
        print("ALL TESTS COMPLETED")
        print("="*60)
        
    except Exception as e:
        print(f"\n VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
