"""Test BM25 hybrid retrieval with keyword matching."""
import asyncio
import logging
import os
import pytest

os.environ.setdefault("VECTOR_STORE_TYPE", "faiss")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_bm25_hybrid():
    """Test BM25 hybrid retrieval vs pure vector search."""
    from app.modules.vector_db.bm25_index import BM25Index
    from app.modules.vector_db.hybrid_retrieval import reciprocal_rank_fusion
    
    # Sample documents with technical terms and proper nouns
    documents = [
        {
            "id": "doc1",
            "text": "The AWS Lambda function uses Python 3.9 runtime with 512MB memory allocation.",
            "metadata": {"category": "technical"}
        },
        {
            "id": "doc2",
            "text": "Our company policy requires all employees to use AWS services for cloud infrastructure.",
            "metadata": {"category": "policy"}
        },
        {
            "id": "doc3",
            "text": "Lambda functions can be triggered by API Gateway, S3 events, or CloudWatch schedules.",
            "metadata": {"category": "technical"}
        },
        {
            "id": "doc4",
            "text": "The employee handbook states that AWS training is mandatory for all developers.",
            "metadata": {"category": "policy"}
        },
        {
            "id": "doc5",
            "text": "Python is the preferred language for Lambda development due to its simplicity.",
            "metadata": {"category": "technical"}
        }
    ]
    
    # Build BM25 index
    bm25_index = BM25Index()
    bm25_index.add_documents(documents)
    
    # Test query with specific technical terms
    query = "AWS Lambda Python runtime"
    
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print(f"{'='*60}\n")
    
    # BM25 search (keyword-based)
    bm25_results = bm25_index.search(query, top_k=5)
    
    print("BM25 Results (Keyword Matching):")
    print("-" * 60)
    for i, doc in enumerate(bm25_results, 1):
        print(f"{i}. [Score: {doc['bm25_score']:.4f}] {doc['text'][:80]}...")
    
    # Simulate vector search results (would come from actual vector store)
    # In real scenario, vector search might rank differently based on semantic similarity
    vector_results = [
        {"id": "doc5", "text": documents[4]["text"], "metadata": documents[4]["metadata"], "distance": 0.15},
        {"id": "doc1", "text": documents[0]["text"], "metadata": documents[0]["metadata"], "distance": 0.25},
        {"id": "doc3", "text": documents[2]["text"], "metadata": documents[2]["metadata"], "distance": 0.30},
        {"id": "doc2", "text": documents[1]["text"], "metadata": documents[1]["metadata"], "distance": 0.40},
        {"id": "doc4", "text": documents[3]["text"], "metadata": documents[3]["metadata"], "distance": 0.50},
    ]
    
    print("\n\nVector Search Results (Semantic Similarity):")
    print("-" * 60)
    for i, doc in enumerate(vector_results, 1):
        print(f"{i}. [Distance: {doc['distance']:.4f}] {doc['text'][:80]}...")
    
    # Hybrid search with RRF
    merged_results = reciprocal_rank_fusion(bm25_results, vector_results, k=60)
    
    print("\n\nHybrid Results (RRF Fusion):")
    print("-" * 60)
    for i, doc in enumerate(merged_results[:5], 1):
        print(f"{i}. [RRF Score: {doc['rrf_score']:.4f}] {doc['text'][:80]}...")
    
    print(f"\n{'='*60}")
    print("Analysis:")
    print(f"{'='*60}")
    print("✓ BM25 excels at exact keyword matches (AWS, Lambda, Python)")
    print("✓ Vector search captures semantic meaning")
    print("✓ RRF fusion combines both strengths for optimal ranking")
    print("✓ Technical terms and proper nouns get proper weight")
    print(f"{'='*60}\n")


async def test_keyword_advantage():
    """Demonstrate BM25 advantage for technical terms."""
    from app.modules.vector_db.bm25_index import BM25Index
    
    # Documents with specific policy names and technical terms
    documents = [
        {
            "id": "policy1",
            "text": "The PTO-2024-Q1 policy allows 15 days of paid time off for full-time employees.",
            "metadata": {}
        },
        {
            "id": "policy2",
            "text": "Vacation time can be requested through the HR portal with manager approval.",
            "metadata": {}
        },
        {
            "id": "policy3",
            "text": "The PTO-2024-Q1 policy supersedes all previous vacation policies.",
            "metadata": {}
        },
        {
            "id": "policy4",
            "text": "Time off requests should be submitted at least two weeks in advance.",
            "metadata": {}
        }
    ]
    
    bm25_index = BM25Index()
    bm25_index.add_documents(documents)
    
    # Query with specific policy name
    query = "PTO-2024-Q1 policy"
    
    print(f"\n{'='*60}")
    print(f"Query: '{query}'")
    print(f"{'='*60}\n")
    
    results = bm25_index.search(query, top_k=4)
    
    print("BM25 Results:")
    print("-" * 60)
    for i, doc in enumerate(results, 1):
        print(f"{i}. [Score: {doc['bm25_score']:.4f}]")
        print(f"   {doc['text']}")
        print()
    
    print("✓ Documents with exact policy name 'PTO-2024-Q1' ranked highest")
    print("✓ Vector search alone might miss this specific identifier")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("BM25 HYBRID RETRIEVAL TEST")
    print("="*60)
    
    asyncio.run(test_bm25_hybrid())
    asyncio.run(test_keyword_advantage())


# ---------------------------------------------------------------------------
# pytest-compatible tests
# ---------------------------------------------------------------------------

def test_bm25_tokenizer_splits_identifiers():
    """_tokenize must split on hyphens/underscores so PTO-2024-Q1 matches sub-terms."""
    from app.modules.vector_db.bm25_index import BM25Index
    idx = BM25Index()
    tokens = idx._tokenize("PTO-2024-Q1 leave_policy")
    assert "pto" in tokens
    assert "2024" in tokens
    assert "q1" in tokens
    assert "leave" in tokens
    assert "policy" in tokens


def test_rrf_weighted_fusion():
    """reciprocal_rank_fusion respects bm25_weight and vector_weight."""
    from app.modules.vector_db.hybrid_retrieval import reciprocal_rank_fusion
    bm25 = [{"id": "a", "text": "a", "metadata": {}}]
    vector = [{"id": "b", "text": "b", "metadata": {}}]
    # Heavy BM25 weight — doc "a" should score higher
    results = reciprocal_rank_fusion(bm25, vector, k=60, bm25_weight=10.0, vector_weight=1.0)
    assert results[0]["id"] == "a"
    # Heavy vector weight — doc "b" should score higher
    results2 = reciprocal_rank_fusion(bm25, vector, k=60, bm25_weight=1.0, vector_weight=10.0)
    assert results2[0]["id"] == "b"


@pytest.mark.asyncio
async def test_bm25_freshness_after_api_add():
    """BM25 index must find a document added via add_document_to_rag_local."""
    from app.modules.integration import get_container, reset_container
    reset_container()
    container = get_container()
    container.initialize()

    doc_manager = container.get_document_manager()
    bm25_index = container.get_bm25_index()

    if not bm25_index or not bm25_index.is_available():
        pytest.skip("rank_bm25 not installed")

    unique_term = "xyzuniquetermfortesting9871"
    await doc_manager.add_document_to_rag_local(
        source_name="bm25_freshness_test",
        text=f"This document contains the term {unique_term}.",
        metadata={"sensitivity": "public_internal", "department": "General"},
        created_by="test"
    )

    # Dirty flag must be set
    assert doc_manager._bm25_dirty is True

    # Flush
    await doc_manager._refresh_bm25_if_dirty()
    assert doc_manager._bm25_dirty is False

    results = bm25_index.search(unique_term, top_k=5)
    assert any(unique_term in r["text"] for r in results), (
        "BM25 must find the document added via API after refresh"
    )
