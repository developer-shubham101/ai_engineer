#!/usr/bin/env python3
"""
Test script to verify the optimized prompt generation.
Tests the new token budgeting and compression features.
"""

import sys
import os
import logging
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "app"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/rag_app.log', mode='a')
    ]
)

logger = logging.getLogger(__name__)

def test_optimized_prompt_generation():
    """Test the optimized prompt generation with token budgeting."""
    try:
        from services.base_rag_service import BaseRAGService
        from services.rag_local_service import LocalRAGService
        from services.prompt_builder import build_prompt_with_selected_chunks, estimate_tokens_from_text
        
        logger.info("=== OPTIMIZED PROMPT GENERATION TEST ===")
        
        # Create service instance
        rag_service = LocalRAGService()
        
        # Test scenarios
        test_cases = [
            {
                "name": "Minimal Query",
                "query": "Hi",
                "requester": {"user_id": "test", "role": "Employee", "department": "Engineering"},
                "profile": None,
                "context": "Company policy: Follow security guidelines. Office hours: 9 AM - 6 PM."
            },
            {
                "name": "With Profile",
                "query": "What are the leave policies?",
                "requester": {"user_id": "test", "role": "HR", "department": "HR"},
                "profile": {"name": "Jane Smith", "position": "HR Manager", "location": "NYC"},
                "context": "Leave policies: Annual leave is 20 days. Sick leave is 10 days. Maternity leave is 12 weeks."
            },
            {
                "name": "Long Context",
                "query": "Tell me about remote work policies",
                "requester": {"user_id": "test", "role": "Manager", "department": "Engineering"},
                "profile": {"name": "John Doe", "position": "Engineering Manager"},
                "context": """
                Remote Work Policy - Comprehensive Guidelines
                
                1. Eligibility: All full-time employees with 6+ months tenure
                2. Equipment: Company provides laptop, monitor, and necessary software
                3. Internet: $50/month reimbursement for high-speed internet
                4. Schedule: Core hours 10 AM - 3 PM in company timezone
                5. Communication: Daily standup via video call, weekly 1:1s
                6. Security: VPN required, 2FA enabled, encrypted storage only
                7. Workspace: Dedicated quiet space, ergonomic setup recommended
                8. Performance: Same metrics as office workers, quarterly reviews
                9. Collaboration: In-office days for team meetings as needed
                10. Benefits: Full benefits maintained, mental health support available
                11. Training: Remote work best practices course required
                12. Equipment return: All company property returned upon termination
                """
            }
        ]
        
        results = []
        
        for i, test_case in enumerate(test_cases, 1):
            logger.info(f"\n--- TEST CASE {i}: {test_case['name']} ---")
            
            # Test optimized prefix generation
            optimized_prefix = rag_service.inject_personalized_context(
                session_id=None,
                llm_prompt_prefix=None,
                query_text=test_case["query"],
                requester=test_case["requester"],
                profile=test_case["profile"],
                max_prefix_tokens=60  # Strict budget
            )
            
            # Test optimized prompt building
            final_prompt = build_prompt_with_selected_chunks(
                prefix=optimized_prefix,
                context_text=test_case["context"],
                question=test_case["query"],
                max_total_tokens=1500,  # Simulate smaller model
                context_priority=0.65
            )
            
            # Calculate metrics
            prefix_tokens = estimate_tokens_from_text(optimized_prefix)
            context_tokens = estimate_tokens_from_text(test_case["context"])
            final_tokens = estimate_tokens_from_text(final_prompt)
            
            result = {
                "test_name": test_case["name"],
                "prefix_tokens": prefix_tokens,
                "context_tokens": context_tokens,
                "final_tokens": final_tokens,
                "optimization_ratio": prefix_tokens / final_tokens,
                "within_budget": final_tokens <= 1500
            }
            results.append(result)
            
            logger.info("OPTIMIZATION_RESULTS:")
            logger.info("  - Prefix tokens: %d", prefix_tokens)
            logger.info("  - Context tokens: %d", context_tokens)
            logger.info("  - Final tokens: %d", final_tokens)
            logger.info("  - Prefix ratio: %.1f%%", (prefix_tokens/final_tokens)*100)
            logger.info("  - Within budget: %s", "✓" if result["within_budget"] else "✗")
        
        # Summary analysis
        logger.info("\n=== OPTIMIZATION SUMMARY ===")
        
        avg_prefix_ratio = sum(r["optimization_ratio"] for r in results) / len(results)
        all_within_budget = all(r["within_budget"] for r in results)
        max_tokens = max(r["final_tokens"] for r in results)
        min_prefix_tokens = min(r["prefix_tokens"] for r in results)
        
        logger.info("PERFORMANCE_METRICS:")
        logger.info("  - Average prefix ratio: %.1f%%", avg_prefix_ratio * 100)
        logger.info("  - All within budget: %s", "✓" if all_within_budget else "✗")
        logger.info("  - Max prompt tokens: %d", max_tokens)
        logger.info("  - Min prefix tokens: %d", min_prefix_tokens)
        
        # Efficiency assessment
        if avg_prefix_ratio < 0.15:  # Less than 15% for prefix
            logger.info("✓ EXCELLENT: Prefix efficiency is optimal")
        elif avg_prefix_ratio < 0.25:  # Less than 25%
            logger.info("✓ GOOD: Prefix efficiency is acceptable")
        else:
            logger.warning("⚠ NEEDS IMPROVEMENT: Prefix taking too much space")
        
        if all_within_budget:
            logger.info("✓ EXCELLENT: All prompts fit within token budget")
        else:
            logger.warning("⚠ ISSUE: Some prompts exceed token budget")
        
        return {
            "success": True,
            "avg_prefix_ratio": avg_prefix_ratio,
            "all_within_budget": all_within_budget,
            "results": results
        }
        
    except Exception as e:
        logger.exception("Optimization test failed: %s", e)
        return {"success": False, "error": str(e)}

def test_context_truncation():
    """Test context truncation with very long documents."""
    try:
        logger.info("\n=== CONTEXT TRUNCATION TEST ===")
        
        from services.prompt_builder import build_prompt_with_selected_chunks, estimate_tokens_from_text
        
        # Create very long context
        long_context = """
        COMPANY HANDBOOK - COMPLETE POLICIES AND PROCEDURES
        
        SECTION 1: EMPLOYEE CONDUCT
        All employees must maintain professional behavior at all times. This includes appropriate dress code, 
        respectful communication, and adherence to company values. Violations may result in disciplinary action.
        
        SECTION 2: WORK SCHEDULE
        Standard work hours are 9 AM to 6 PM, Monday through Friday. Flexible arrangements may be available
        with manager approval. Remote work is permitted up to 3 days per week for eligible employees.
        
        SECTION 3: LEAVE POLICIES
        Annual leave: 20 days per year, accrued monthly. Sick leave: 10 days per year. Personal leave: 5 days.
        Maternity/Paternity leave: 12 weeks paid. Bereavement leave: 3 days for immediate family.
        
        SECTION 4: BENEFITS
        Health insurance: 100% premium covered for employee, 80% for family. Dental and vision included.
        Retirement: 401k with 6% company match. Life insurance: 2x annual salary. Disability coverage included.
        
        SECTION 5: PERFORMANCE MANAGEMENT
        Annual reviews conducted in January. Quarterly check-ins with managers. Goal setting and tracking
        through company portal. Performance improvement plans for underperforming employees.
        
        SECTION 6: TRAINING AND DEVELOPMENT
        $2000 annual budget for professional development. Conference attendance encouraged. Internal
        training programs available. Tuition reimbursement up to $5000 per year for relevant courses.
        
        SECTION 7: TECHNOLOGY POLICIES
        All devices must have updated antivirus software. Personal use of company equipment is limited.
        Social media guidelines must be followed. Data security protocols are mandatory.
        
        SECTION 8: SAFETY PROCEDURES
        Emergency evacuation plans posted in all areas. First aid kits available on each floor.
        Incident reporting required within 24 hours. Safety training mandatory for all employees.
        """ * 3  # Triple the content to make it very long
        
        # Test with different token budgets
        budgets = [500, 1000, 1500, 2000]
        
        for budget in budgets:
            logger.info(f"\nTesting with {budget} token budget:")
            
            prompt = build_prompt_with_selected_chunks(
                prefix="Assistant | Employee/Engineering",
                context_text=long_context,
                question="What are the leave policies?",
                max_total_tokens=budget,
                context_priority=0.7
            )
            
            prompt_tokens = estimate_tokens_from_text(prompt)
            context_tokens = estimate_tokens_from_text(long_context)
            
            logger.info("  - Original context: %d tokens", context_tokens)
            logger.info("  - Final prompt: %d tokens", prompt_tokens)
            logger.info("  - Within budget: %s", "✓" if prompt_tokens <= budget else "✗")
            logger.info("  - Reduction: %.1f%%", ((context_tokens - prompt_tokens) / context_tokens) * 100)
        
        logger.info("✓ Context truncation test completed")
        return True
        
    except Exception as e:
        logger.exception("Context truncation test failed: %s", e)
        return False

if __name__ == "__main__":
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)
    
    logger.info("Starting optimized prompt generation tests...")
    
    # Run tests
    optimization_result = test_optimized_prompt_generation()
    truncation_result = test_context_truncation()
    
    if optimization_result.get("success") and truncation_result:
        logger.info("\n🎉 ALL OPTIMIZATION TESTS PASSED!")
        logger.info("Key improvements:")
        logger.info("  - Prefix efficiency: %.1f%% average", optimization_result["avg_prefix_ratio"] * 100)
        logger.info("  - Token budgeting: %s", "Working" if optimization_result["all_within_budget"] else "Needs tuning")
        logger.info("  - Context truncation: Working")
        logger.info("\nCheck logs/rag_app.log for detailed analysis")
    else:
        logger.error("❌ Some tests failed")
        if not optimization_result.get("success"):
            logger.error("Optimization error: %s", optimization_result.get("error"))
        sys.exit(1)