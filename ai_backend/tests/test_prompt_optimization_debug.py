#!/usr/bin/env python3
"""
Test script to debug prompt optimization in the RAG system.
This script tests the prompt building process without requiring the full server to be running.
"""

import sys
import os
import logging
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

# Configure logging to see debug output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/rag_app.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

def test_prompt_optimization():
    """Test the prompt optimization process."""
    try:
        # Import required services
        from services.base_rag_service import BaseRAGService
        from services.rag_local_service import LocalRAGService
        from services.prompt_builder import build_prompt_with_selected_chunks, estimate_tokens_from_text
        
        logger.info("=== PROMPT OPTIMIZATION DEBUG TEST ===")
        
        # Create a local RAG service instance
        rag_service = LocalRAGService()
        
        # Test data
        query_text = "Hi"
        requester = {
            "user_id": "test_user",
            "role": "Employee", 
            "department": "Engineering"
        }
        
        # Test 1: Basic prompt optimization
        logger.info("\n--- TEST 1: Basic Prompt Optimization ---")
        
        optimized_prefix = rag_service.inject_personalized_context(
            session_id=None,
            llm_prompt_prefix=None,
            query_text=query_text,
            requester=requester,
            profile=None
        )
        
        # Test 2: Prompt building with context
        logger.info("\n--- TEST 2: Prompt Building with Context ---")
        
        # Simulate some context documents
        context_text = "Company policy states that employees should follow security guidelines. Our office hours are 9 AM to 6 PM."
        
        final_prompt = build_prompt_with_selected_chunks(
            prefix=optimized_prefix,
            context_text=context_text,
            question=query_text
        )
        
        # Test 3: Token efficiency analysis
        logger.info("\n--- TEST 3: Token Efficiency Analysis ---")
        
        prefix_tokens = estimate_tokens_from_text(optimized_prefix)
        context_tokens = estimate_tokens_from_text(context_text)
        query_tokens = estimate_tokens_from_text(query_text)
        total_tokens = estimate_tokens_from_text(final_prompt)
        
        logger.info("TOKEN_EFFICIENCY_REPORT:")
        logger.info("  - Prefix tokens: %d (%.1f%%)", prefix_tokens, (prefix_tokens/total_tokens)*100)
        logger.info("  - Context tokens: %d (%.1f%%)", context_tokens, (context_tokens/total_tokens)*100)
        logger.info("  - Query tokens: %d (%.1f%%)", query_tokens, (query_tokens/total_tokens)*100)
        logger.info("  - Total tokens: %d", total_tokens)
        
        # Test 4: Different user profiles
        logger.info("\n--- TEST 4: Profile-based Optimization ---")
        
        profile = {
            "name": "John Doe",
            "position": "Senior Developer",
            "location": "New York"
        }
        
        profile_prefix = rag_service.inject_personalized_context(
            session_id=None,
            llm_prompt_prefix=None,
            query_text=query_text,
            requester=requester,
            profile=profile
        )
        
        profile_prompt = build_prompt_with_selected_chunks(
            prefix=profile_prefix,
            context_text=context_text,
            question=query_text
        )
        
        profile_tokens = estimate_tokens_from_text(profile_prompt)
        
        logger.info("PROFILE_COMPARISON:")
        logger.info("  - Basic prompt tokens: %d", total_tokens)
        logger.info("  - Profile prompt tokens: %d", profile_tokens)
        logger.info("  - Token difference: %d", profile_tokens - total_tokens)
        
        # Test 5: Long context optimization
        logger.info("\n--- TEST 5: Long Context Optimization ---")
        
        long_context = """
        Our company Saarthi Infotech Pvt. Ltd. has comprehensive policies for employee management.
        The HR department handles all employee-related queries and maintains strict confidentiality.
        Security policies require all employees to use strong passwords and enable two-factor authentication.
        The IT department provides technical support during business hours from 9 AM to 6 PM.
        Financial policies require proper documentation for all expense reimbursements.
        Legal compliance is maintained through regular audits and policy updates.
        Employee benefits include health insurance, retirement plans, and professional development opportunities.
        Remote work policies allow flexible arrangements with manager approval.
        Performance reviews are conducted quarterly with clear metrics and feedback.
        Training programs are available for skill development and career advancement.
        """
        
        long_prompt = build_prompt_with_selected_chunks(
            prefix=optimized_prefix,
            context_text=long_context,
            question="What are the company policies for remote work?"
        )
        
        long_tokens = estimate_tokens_from_text(long_prompt)
        
        logger.info("LONG_CONTEXT_ANALYSIS:")
        logger.info("  - Long context tokens: %d", estimate_tokens_from_text(long_context))
        logger.info("  - Long prompt total tokens: %d", long_tokens)
        
        if long_tokens > 1500:
            logger.warning("OPTIMIZATION_NEEDED: Prompt exceeds 1500 tokens, consider chunking")
        
        logger.info("\n=== PROMPT OPTIMIZATION TEST COMPLETE ===")
        
        return {
            "basic_tokens": total_tokens,
            "profile_tokens": profile_tokens,
            "long_tokens": long_tokens,
            "optimization_successful": True
        }
        
    except Exception as e:
        logger.exception("Test failed: %s", e)
        return {"optimization_successful": False, "error": str(e)}

def test_rbac_filtering():
    """Test RBAC filtering without LLM calls."""
    try:
        logger.info("\n--- RBAC FILTERING TEST ---")
        
        from services.base_rag_service import BaseRAGService
        
        # Create base service instance
        base_service = BaseRAGService()
        
        # Test documents with different sensitivity levels
        test_docs = [
            "Public company information available to all employees.",
            "HR confidential information about employee benefits.",
            "Highly confidential executive strategy document."
        ]
        
        test_metadatas = [
            {"sensitivity": "public_internal", "department": "General"},
            {"sensitivity": "role_confidential", "department": "HR"},
            {"sensitivity": "highly_confidential", "department": "Executive"}
        ]
        
        test_ids = ["doc_1", "doc_2", "doc_3"]
        test_distances = [0.1, 0.2, 0.3]
        
        # Test different user roles
        test_users = [
            {"user_id": "guest", "role": "Guest", "department": "General"},
            {"user_id": "employee", "role": "Employee", "department": "Engineering"},
            {"user_id": "hr", "role": "HR", "department": "HR"},
            {"user_id": "admin", "role": "SuperAdmin", "department": "Executive"}
        ]
        
        for user in test_users:
            logger.info("Testing RBAC for user: %s (role: %s)", user["user_id"], user["role"])
            
            filtered_result = base_service.filter_documents_by_rbac(
                raw_docs=test_docs,
                raw_metadatas=test_metadatas,
                raw_ids=test_ids,
                raw_distances=test_distances,
                requester=user
            )
            
            visible_count = len(filtered_result["documents"])
            filtered_count = filtered_result["filtered_out_count"]
            
            logger.info("  - Visible documents: %d", visible_count)
            logger.info("  - Filtered documents: %d", filtered_count)
            
        logger.info("RBAC filtering test complete")
        
    except Exception as e:
        logger.exception("RBAC test failed: %s", e)

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    logger.info("Starting prompt optimization debug test...")
    
    # Run tests
    result = test_prompt_optimization()
    test_rbac_filtering()
    
    if result.get("optimization_successful"):
        logger.info("✅ All tests completed successfully!")
        logger.info("Check logs/rag_app.log for detailed debug information")
    else:
        logger.error("❌ Tests failed: %s", result.get("error"))
        sys.exit(1)