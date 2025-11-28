#!/usr/bin/env python3
"""
Test script to show optimized prompt with conversation context.
"""
import asyncio
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.base_rag_service import BaseRAGService
from app.services.rag_local_service import LocalRAGService

# Mock the fetch_recent_messages function
def mock_fetch_recent_messages(session_id, limit=4):
    return [
        {
            "speaker": "user",
            "content": "What is our company policy on remote work?",
            "tone": "polite"
        },
        {
            "speaker": "assistant", 
            "content": "Our company allows 3 days per week remote work with home office stipend."
        },
        {
            "speaker": "user",
            "content": "Hi",
            "tone": "polite"
        }
    ]

async def test_conversation_context():
    """Test the optimized prompt with conversation context."""
    
    # Patch the fetch function
    import app.services.base_rag_service
    app.services.base_rag_service.fetch_recent_messages = mock_fetch_recent_messages
    
    # Create a local RAG service instance
    service = LocalRAGService()
    
    # Mock requester data
    requester = {
        "user_id": "u_admin_1",
        "role": "SuperAdmin", 
        "department": "Executive"
    }
    
    # Mock profile data
    profile = {
        "name": "Admin User",
        "location": "HQ",
        "position": "Administrator"
    }
    
    # Test the optimized prompt building with conversation context
    session_id = "sess_test123"
    query_text = "Can you tell me more about the stipend?"
    llm_prompt_prefix = None
    
    print("Testing optimized prompt with conversation context...")
    
    # Call the inject_personalized_context method directly
    optimized_prefix = service.inject_personalized_context(
        session_id=session_id,
        llm_prompt_prefix=llm_prompt_prefix,
        query_text=query_text,
        requester=requester,
        profile=profile
    )
    
    print("=== OPTIMIZED PROMPT WITH CONVERSATION CONTEXT ===")
    print(optimized_prefix)
    print("=== END ===")
    
    print(f"Prompt length: {len(optimized_prefix)} characters")
    print(f"Estimated tokens: {len(optimized_prefix) // 4}")

if __name__ == "__main__":
    asyncio.run(test_conversation_context())