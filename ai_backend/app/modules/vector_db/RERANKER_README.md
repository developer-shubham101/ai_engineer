# Cross-Encoder Reranking System

## Overview

This module implements **cross-encoder reranking** for improved retrieval quality in RAG systems. This is a **Tier 1 optimization** - the single highest-impact change you can make to improve RAG performance.

## Why Reranking?

**The Problem**: Vector similarity (cosine distance) is fast but imperfect. It can rank less relevant documents higher than more relevant ones because embeddings capture semantic similarity in a compressed space.

**The Solution**: Cross-encoders score query-document pairs directly, capturing nuanced relevance that embeddings miss.

## Architecture

```
Query → Vector Search (top-20) → RBAC Filter → Cross-Encoder Rerank → Top-3 to LLM
```

### Pipeline Steps

1. **Over-fetch**: Retrieve 20 documents from vector store (4x more than needed)
2. **Filter**: Apply RBAC and metadata filtering
3. **Rerank**: Score each query-document pair with cross-encoder
4. **Select**: Return top-K highest scoring documents

## Usage

### Basic Usage

```python
from app.modules.vector_db.reranker import CrossEncoderReranker

# Initialize reranker
reranker = CrossEncoderReranker()

# Rerank documents
reranked = reranker.rerank(
    query="What is the vacation policy?",
    documents=retrieved_docs,  # List of dicts with 'text' field
    top_k=3
)

# Access reranked results
for doc in reranked:
    print(f"Score: {doc['rerank_score']:.4f}")
    print(f"Text: {doc['text']}")
```

### Custom Model

```python
# Use a different cross-encoder model
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-TinyBERT-L-6"
)
```

### Integration with RAG

Reranking is **automatically enabled** in the RAG orchestrator:

```python
# In RAGOrchestrator.retrieve_documents()
# 1. Retrieves top-20 from vector store
# 2. Applies RBAC filtering
# 3. Reranks with cross-encoder
# 4. Returns top-K to LLM
```

No code changes needed - just use the RAG API as normal!

## Model Details

### Default Model: cross-encoder/ms-marco-MiniLM-L6-v2

- **Type**: Cross-encoder for passage ranking
- **Training**: MS MARCO passage ranking dataset
- **Parameters**: ~90M
- **Size**: ~400MB
- **Speed**: ~5-10ms per query-document pair (CPU)
- **Accuracy**: State-of-art for passage reranking

### How Cross-Encoders Work

Unlike bi-encoders (used for embeddings), cross-encoders:
1. Concatenate query and document: `[CLS] query [SEP] document [SEP]`
2. Process through transformer
3. Output relevance score directly

This allows attention between query and document tokens, capturing nuanced relevance.

## Performance

### Latency

- **20 documents**: ~100-200ms (CPU)
- **50 documents**: ~250-500ms (CPU)
- **GPU**: 5-10x faster

### Memory

- **Model**: ~400MB
- **Runtime**: ~100MB additional

### Quality Improvement

Typical improvements over vector similarity alone:
- **Precision@3**: +15-25%
- **NDCG@10**: +10-20%
- **User satisfaction**: Significant improvement

## Testing

### Run Test Script

```bash
python test_reranker.py
```

### Expected Output

```
Testing Cross-Encoder Reranker
============================================================

1. Initializing reranker...
2. Query: 'What is the company vacation policy?'

3. Original documents (by vector similarity):
   1. [distance=0.150] Our company offers 20 days of paid vacation per year...
   2. [distance=0.180] The office cafeteria serves lunch from 12pm to 2pm...
   3. [distance=0.200] Vacation days must be requested at least 2 weeks...

4. Reranking with cross-encoder...

5. Reranked documents (top-3):
   1. [rerank_score=8.2341, original_distance=0.150] Our company offers 20 days...
   2. [rerank_score=7.8923, original_distance=0.200] Vacation days must be requested...
   3. [rerank_score=7.5612, original_distance=0.220] Employees can carry over up to 5...

6. Analysis:
   - Original top-3 IDs: ['doc1', 'doc2', 'doc3']
   - Reranked top-3 IDs: ['doc1', 'doc3', 'doc4']
   - Original relevant docs in top-3: 2
   - Reranked relevant docs in top-3: 3
   ✅ Reranking improved results!
```

## Configuration

### Environment Variables

No environment variables needed - works out of the box!

### Tuning Parameters

```python
# Adjust over-fetch ratio in RAGOrchestrator
retrieval_k = max(top_k * 4, 20)  # 4x or minimum 20

# Adjust in your code
reranker.rerank(query, documents, top_k=5)  # Return top 5 instead of 3
```

## Benefits

### Why This is Tier 1

1. **Highest ROI**: Maximum impact for minimal code
2. **Proven approach**: Industry-standard technique
3. **Easy to validate**: Clear before/after metrics
4. **No prompt changes**: Fixes data quality at source
5. **Transparent**: Works with existing API

### Comparison to Other Optimizations

| Optimization | Impact | Effort | ROI |
|--------------|--------|--------|-----|
| **Reranking** | High | Low | **Highest** |
| Prompt tuning | Medium | Medium | Medium |
| Model selection | Medium | Low | Medium |
| Fine-tuning | High | High | Low |

## Troubleshooting

### Model Download Issues

```python
# Model downloads automatically on first use
# If download fails, manually download:
from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2")
```

### Performance Issues

```python
# Use smaller model for faster inference
reranker = CrossEncoderReranker(
    model_name="cross-encoder/ms-marco-TinyBERT-L-6"
)
```

### Fallback Behavior

If reranking fails, the system automatically falls back to vector similarity:

```python
try:
    reranked = reranker.rerank(query, documents, top_k)
except Exception as e:
    logger.warning(f"Reranking failed: {e}")
    return documents[:top_k]  # Fallback to vector similarity
```

## Future Enhancements

- [ ] Caching of reranking scores
- [ ] Batch processing for multiple queries
- [ ] GPU acceleration support
- [ ] Hybrid scoring (vector + rerank)
- [ ] A/B testing framework

## References

- [MS MARCO Dataset](https://microsoft.github.io/msmarco/)
- [Sentence Transformers](https://www.sbert.net/)
- [Cross-Encoders for Ranking](https://www.sbert.net/examples/applications/cross-encoder/README.html)

## License

Same as parent project (MIT)
