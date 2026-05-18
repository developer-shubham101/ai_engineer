"""Test script for cross-encoder reranker."""
import asyncio
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.modules.vector_db.reranker import CrossEncoderReranker


async def test_reranker():
    """Test the cross-encoder reranker."""
    print("=" * 60)
    print("Testing Cross-Encoder Reranker")
    print("=" * 60)
    
    # Initialize reranker
    print("\n1. Initializing reranker...")
    reranker = CrossEncoderReranker()
    
    # Test query
    query = "What is the company vacation policy?"
    
    # Mock documents (simulating vector retrieval results)
    documents = [
        {
            "id": "doc1",
            "text": "Our company offers 20 days of paid vacation per year for all full-time employees.",
            "metadata": {"source": "HR Policy"},
            "distance": 0.15
        },
        {
            "id": "doc2",
            "text": "The office cafeteria serves lunch from 12pm to 2pm daily.",
            "metadata": {"source": "Office Guide"},
            "distance": 0.18
        },
        {
            "id": "doc3",
            "text": "Vacation days must be requested at least 2 weeks in advance through the HR portal.",
            "metadata": {"source": "HR Policy"},
            "distance": 0.20
        },
        {
            "id": "doc4",
            "text": "Employees can carry over up to 5 unused vacation days to the next year.",
            "metadata": {"source": "HR Policy"},
            "distance": 0.22
        },
        {
            "id": "doc5",
            "text": "The company parking lot is available for all employees with valid badges.",
            "metadata": {"source": "Office Guide"},
            "distance": 0.25
        }
    ]
    
    print(f"\n2. Query: '{query}'")
    print(f"\n3. Original documents (by vector similarity):")
    for i, doc in enumerate(documents, 1):
        print(f"   {i}. [distance={doc['distance']:.3f}] {doc['text'][:80]}...")
    
    # Rerank documents
    print(f"\n4. Reranking with cross-encoder...")
    reranked = reranker.rerank(query, documents, top_k=3)
    
    print(f"\n5. Reranked documents (top-3):")
    for i, doc in enumerate(reranked, 1):
        print(f"   {i}. [rerank_score={doc['rerank_score']:.4f}, original_distance={doc['original_distance']:.3f}]")
        print(f"      {doc['text'][:80]}...")
    
    # Show improvement
    print(f"\n6. Analysis:")
    print(f"   - Original top-3 IDs: {[d['id'] for d in documents[:3]]}")
    print(f"   - Reranked top-3 IDs: {[d['id'] for d in reranked]}")
    
    # Check if reranking improved results
    original_relevant = sum(1 for d in documents[:3] if "vacation" in d['text'].lower())
    reranked_relevant = sum(1 for d in reranked if "vacation" in d['text'].lower())
    
    print(f"   - Original relevant docs in top-3: {original_relevant}")
    print(f"   - Reranked relevant docs in top-3: {reranked_relevant}")
    
    if reranked_relevant > original_relevant:
        print(f"   ✅ Reranking improved results!")
    elif reranked_relevant == original_relevant:
        print(f"   ✓ Reranking maintained quality")
    else:
        print(f"   ⚠️ Reranking may need tuning")
    
    # Model info
    print(f"\n7. Model info:")
    info = reranker.get_model_info()
    for key, value in info.items():
        print(f"   - {key}: {value}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_reranker())
