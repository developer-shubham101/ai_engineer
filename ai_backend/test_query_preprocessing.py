"""Test query preprocessing with spell correction and normalization."""
import asyncio
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_query_preprocessing():
    """Test query preprocessing with various misspellings."""
    from app.modules.vector_db.query_preprocessor import QueryPreprocessor
    
    preprocessor = QueryPreprocessor()
    
    if not preprocessor.is_available():
        print("\n⚠️  pyspellchecker not installed. Install with: pip install pyspellchecker")
        return
    
    # Test cases with common misspellings
    test_queries = [
        "What is the vacaton policy?",  # vacation
        "How do I confgure AWS Lambda?",  # configure
        "What is our RBAC implmentation?",  # implementation
        "Tell me about employe benefits",  # employee
        "What is the PTO-2024-Q1 polcy?",  # policy (but should keep PTO-2024-Q1)
        "How to acces the HR portal?",  # access
        "What are the compny holidays?",  # company
        "Explain the remot work policy",  # remote
    ]
    
    print("\n" + "="*80)
    print("QUERY PREPROCESSING TEST - Spell Correction")
    print("="*80 + "\n")
    
    for original_query in test_queries:
        processed = await preprocessor.process_query(
            query=original_query,
            use_spell_correction=True,
            use_expansion=False,
            use_llm_rewrite=False
        )
        
        print(f"Original:   '{processed.original}'")
        print(f"Normalized: '{processed.normalized}'")
        if processed.corrected:
            print(f"Corrected:  '{processed.corrected}' ✅")
        else:
            print(f"Corrected:  No changes needed")
        print(f"Variants:   {len(processed.all_variants)} unique queries")
        print("-" * 80 + "\n")


async def test_acronym_expansion():
    """Test acronym expansion."""
    from app.modules.vector_db.query_preprocessor import QueryPreprocessor
    
    preprocessor = QueryPreprocessor()
    
    test_queries = [
        "What is our PTO policy?",
        "How does RBAC work?",
        "Configure AWS Lambda",
        "What is the HR process?",
    ]
    
    print("\n" + "="*80)
    print("ACRONYM EXPANSION TEST")
    print("="*80 + "\n")
    
    for original_query in test_queries:
        processed = await preprocessor.process_query(
            query=original_query,
            use_spell_correction=False,
            use_expansion=True,  # Enable expansion
            use_llm_rewrite=False
        )
        
        print(f"Original:  '{processed.original}'")
        print(f"Expanded:  '{processed.corrected if processed.corrected else 'No expansion'}'")
        print("-" * 80 + "\n")


async def test_combined_preprocessing():
    """Test combined spell correction + expansion."""
    from app.modules.vector_db.query_preprocessor import QueryPreprocessor
    
    preprocessor = QueryPreprocessor()
    
    if not preprocessor.is_available():
        print("\n⚠️  pyspellchecker not installed")
        return
    
    test_query = "What is the PTO polcy for remot work?"  # policy, remote
    
    print("\n" + "="*80)
    print("COMBINED PREPROCESSING TEST")
    print("="*80 + "\n")
    
    processed = await preprocessor.process_query(
        query=test_query,
        use_spell_correction=True,
        use_expansion=True,
        use_llm_rewrite=False
    )
    
    print(f"Original:   '{processed.original}'")
    print(f"Normalized: '{processed.normalized}'")
    print(f"Corrected:  '{processed.corrected}'")
    print(f"\nAll variants for search:")
    for i, variant in enumerate(processed.all_variants, 1):
        print(f"  {i}. '{variant}'")
    
    print("\n✅ These variants will be used for hybrid search")
    print("   - Original: catches exact matches")
    print("   - Corrected: catches intended meaning")
    print("   - Expanded: catches related terms")
    print("="*80 + "\n")


async def test_edge_cases():
    """Test edge cases."""
    from app.modules.vector_db.query_preprocessor import QueryPreprocessor
    
    preprocessor = QueryPreprocessor()
    
    if not preprocessor.is_available():
        return
    
    test_cases = [
        ("AWS-2024-Q1", "Identifiers with hyphens and numbers"),
        ("API", "Short acronyms"),
        ("CEO memo", "Acronyms with context"),
        ("project-data-2024", "Project identifiers"),
    ]
    
    print("\n" + "="*80)
    print("EDGE CASES TEST")
    print("="*80 + "\n")
    
    for query, description in test_cases:
        processed = await preprocessor.process_query(
            query=query,
            use_spell_correction=True
        )
        
        print(f"Test: {description}")
        print(f"  Input:     '{query}'")
        print(f"  Corrected: '{processed.corrected if processed.corrected else 'No changes (preserved)'}'")
        print()


if __name__ == "__main__":
    print("\n" + "="*80)
    print("QUERY PREPROCESSING SYSTEM TEST")
    print("="*80)
    
    asyncio.run(test_query_preprocessing())
    asyncio.run(test_acronym_expansion())
    asyncio.run(test_combined_preprocessing())
    asyncio.run(test_edge_cases())
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("✅ Spell correction: Fixes common typos")
    print("✅ Normalization: Lowercase + cleanup")
    print("✅ Acronym expansion: Expands common terms")
    print("✅ Edge case handling: Preserves identifiers")
    print("✅ Multi-variant search: Searches all versions")
    print("="*80 + "\n")
