#!/usr/bin/env python3
"""
Test script to verify prompt optimization is working.
"""
import asyncio
import sys
import os

# Add the app directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.services.base_rag_service import BaseRAGService
from app.services.rag_local_service import LocalRAGService

async def test_prompt_optimization():
    """Test the optimized prompt generation."""
    
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
        "gender": "Other"
    }
    
    # Test the optimized prompt building
    session_id = "sess_test123"
    query_text = "Hi"
    llm_prompt_prefix = None
    
    print("Testing optimized prompt generation...")
    
    # Call the inject_personalized_context method directly
    optimized_prefix = service.inject_personalized_context(
        session_id=session_id,
        llm_prompt_prefix=llm_prompt_prefix,
        query_text=query_text,
        requester=requester,
        profile=profile
    )
    
    print("=== OPTIMIZED PROMPT PREFIX ===")
    print(optimized_prefix)
    print("=== END ===")
    
    print(f"Prompt length: {len(optimized_prefix)} characters")
    print(f"Estimated tokens: {len(optimized_prefix) // 4}")

if __name__ == "__main__":
    asyncio.run(test_prompt_optimization())