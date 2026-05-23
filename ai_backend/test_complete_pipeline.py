"""Test complete query preprocessing pipeline."""
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.modules.vector_db.query_preprocessor import QueryPreprocessor, QueryType


async def test_pipeline():
    """Test complete query preprocessing pipeline."""
    preprocessor = QueryPreprocessor()
    
    print("=" * 80)
    print("QUERY PREPROCESSING PIPELINE TEST")
    print("=" * 80)
    print()
    
    # Test cases
    test_queries = [
        # Typos
        ("What is the vacaton polcy?", "Spell correction"),
        ("How do I confgure AWS Lamba?", "Multiple typos"),
        
        # Acronyms
        ("What is our PTO policy?", "Acronym expansion"),
        ("How to setup SSO with MFA?", "Multiple acronyms"),
        
        # Query classification
        ("Who is the CEO?", "Factual query"),
        ("How to submit a leave request?", "Procedural query"),
        ("What is the remote work policy?", "Policy query"),
        ("What is RBAC?", "Definition query"),
        ("Difference between PTO and sick leave?", "Comparison query"),
        ("API error 500 fix", "Troubleshooting query"),
        
        # Combined
        ("What is the PTO polcy for remot work?", "Typos + acronym"),
    ]
    
    for query, description in test_queries:
        print(f"📝 Test: {description}")
        print(f"   Original: '{query}'")
        print()
        
        # Process without expansion
        processed = await preprocessor.process_query(
            query=query,
            use_spell_correction=True,
            use_expansion=False
        )
        
        print(f"   ✅ Query Type: {processed.query_type.value}")
        print(f"   ✅ Normalized: '{processed.normalized}'")
        if processed.corrected:
            print(f"   ✅ Corrected: '{processed.corrected}'")
        
        # Test with expansion
        processed_expanded = await preprocessor.process_query(
            query=query,
            use_spell_correction=True,
            use_expansion=True
        )
        
        if processed_expanded.expanded:
            print(f"   ✅ Expanded: '{processed_expanded.expanded[:100]}...'")
        
        print(f"   📊 Total Variants: {len(processed_expanded.all_variants)}")
        for i, variant in enumerate(processed_expanded.all_variants, 1):
            print(f"      {i}. {variant[:80]}...")
        
        print()
        print("-" * 80)
        print()
    
    # Test availability
    print("🔧 System Status:")
    print(f"   Spell Checker: {'✅ Available' if preprocessor.is_available() else '❌ Not Available'}")
    print(f"   Acronym Dictionary: {len(preprocessor.expansions)} entries")
    print(f"   Query Patterns: {len(preprocessor.query_patterns)} types")
    print()
    
    # Test query classification
    print("🎯 Query Classification Examples:")
    classification_tests = [
        "Who is the manager?",
        "How to reset password?",
        "What is the vacation policy?",
        "What is API?",
        "Compare AWS vs Azure",
        "Fix login error",
        "Tell me about the company"
    ]
    
    for test_query in classification_tests:
        query_type = preprocessor.classify_query(test_query)
        print(f"   '{test_query}' → {query_type.value}")
    
    print()
    print("=" * 80)
    print("✅ PIPELINE TEST COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_pipeline())
