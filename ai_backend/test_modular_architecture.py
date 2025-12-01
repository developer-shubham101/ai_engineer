"""Test script demonstrating the new modular architecture."""

import asyncio
import sys
from pathlib import Path

# Add the app directory to Python path
sys.path.append(str(Path(__file__).parent / "app"))

from app.modules.integration import get_container
from app.modules.api.models import QueryRequest
from app.modules.llm.interfaces import RAGRequest


async def test_modular_architecture():
    """Test the modular architecture with mock data."""
    print("🚀 Testing Modular Architecture")
    print("=" * 50)
    
    # Get the dependency injection container
    container = get_container()
    container.initialize()
    
    print("✅ Container initialized successfully")
    
    # Test 1: Authentication Module
    print("\n1. Testing Authentication Module")
    print("-" * 30)
    
    user_manager = container.get_user_manager()
    authenticator = container.get_authenticator()
    
    # Test user authentication
    user = await user_manager.get_user_by_username("admin")
    if user:
        print(f"✅ Found user: {user['username']} (Role: {user['role']})")
        
        # Test token creation
        token = await authenticator.create_token(user)
        print(f"✅ Created JWT token: {token[:50]}...")
        
        # Test token verification
        payload = await authenticator.verify_token(token)
        if payload:
            print(f"✅ Token verified: {payload['username']}")
    
    # Test 2: Vector Database Module
    print("\n2. Testing Vector Database Module")
    print("-" * 30)
    
    vector_store = container.get_vector_store()
    
    # Test adding a document
    test_doc_id = await vector_store.add_document(
        "This is a test document about company policies.",
        {
            "source": "test",
            "department": "HR",
            "sensitivity": "public_internal"
        }
    )
    print(f"✅ Added test document: {test_doc_id}")
    
    # Test searching documents
    search_results = await vector_store.search_documents(
        "company policies", top_k=3
    )
    print(f"✅ Found {len(search_results)} documents in search")
    
    # Test 3: RBAC Module
    print("\n3. Testing RBAC Module")
    print("-" * 30)
    
    rbac_manager = container.get_rbac_manager()
    
    # Test permission checking
    can_read = await rbac_manager.check_permission(user, "documents", "read")
    can_delete = await rbac_manager.check_permission(user, "documents", "delete")
    
    print(f"✅ User can read documents: {can_read}")
    print(f"✅ User can delete documents: {can_delete}")
    
    # Test document filtering
    filtered_docs = await rbac_manager.filter_documents(search_results, user)
    print(f"✅ Filtered {len(search_results)} docs to {len(filtered_docs)} for user")
    
    # Test 4: Session Management
    print("\n4. Testing Session Management")
    print("-" * 30)
    
    session_manager = container.get_session_manager()
    
    # Create a session
    session_id = await session_manager.create_session(
        user["user_id"], 
        {"role": user["role"], "department": user["department"]}
    )
    print(f"✅ Created session: {session_id}")
    
    # Store messages
    await session_manager.store_message(session_id, "user", "Hello, what are our policies?")
    await session_manager.store_message(session_id, "assistant", "Here are our company policies...")
    
    # Retrieve messages
    messages = await session_manager.get_messages(session_id, limit=5)
    print(f"✅ Retrieved {len(messages)} messages from session")
    
    # Test 5: RAG Orchestration
    print("\n5. Testing RAG Orchestration")
    print("-" * 30)
    
    rag_orchestrator = container.get_rag_orchestrator()
    
    # Create RAG request
    rag_request = RAGRequest(
        question="What are our company policies?",
        user=user,
        session_id=session_id,
        top_k=3,
        use_llm=True,
        debug=True
    )
    
    # Process query
    response = await rag_orchestrator.process_query(rag_request)
    
    print(f"✅ RAG query processed")
    print(f"   - Retrieved {len(response.retrieved_documents)} documents")
    print(f"   - Generated answer: {response.answer is not None}")
    print(f"   - Context length: {len(response.context or '')}")
    
    # Test 6: Document Management
    print("\n6. Testing Document Management")
    print("-" * 30)
    
    document_manager = container.get_document_manager()
    
    # Add a document through document manager
    doc_id = await document_manager.add_document(
        "This is a new HR policy document.",
        {
            "source": "HR Manual",
            "department": "HR",
            "sensitivity": "department_confidential",
            "category": "policy"
        },
        user
    )
    print(f"✅ Added document via document manager: {doc_id}")
    
    # List documents
    user_docs = await document_manager.list_documents(user)
    print(f"✅ User can access {len(user_docs)} documents")
    
    # Test 7: Profile Analysis
    print("\n7. Testing Profile Analysis")
    print("-" * 30)
    
    profile_analyzer = container.get_profile_analyzer()
    
    # Analyze user profile
    profile_analysis = await profile_analyzer.analyze_user_profile(user["user_id"])
    print(f"✅ Profile analysis completed")
    print(f"   - Role: {profile_analysis.get('role')}")
    print(f"   - Department: {profile_analysis.get('department')}")
    print(f"   - Profile completeness: {profile_analysis.get('profile_completeness', 0):.2f}")
    
    # Get personalization context
    context = await profile_analyzer.get_personalization_context(user["user_id"], session_id)
    print(f"✅ Personalization context: {context}")
    
    print("\n" + "=" * 50)
    print("🎉 All modular architecture tests completed successfully!")
    print("\nKey Benefits Demonstrated:")
    print("- ✅ Clean separation of concerns")
    print("- ✅ Interface-based design")
    print("- ✅ Dependency injection")
    print("- ✅ Easy testing and mocking")
    print("- ✅ Swappable implementations")


if __name__ == "__main__":
    asyncio.run(test_modular_architecture())