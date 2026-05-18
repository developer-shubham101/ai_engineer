"""Hybrid retrieval combining BM25 and vector search with RRF fusion."""
from __future__ import annotations

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Merge BM25 and vector results using Reciprocal Rank Fusion.
    
    RRF formula: score = sum(1 / (k + rank)) for each result list
    
    Args:
        bm25_results: Results from BM25 search with 'id' field
        vector_results: Results from vector search with 'id' field
        k: RRF constant (default 60, standard value)
    
    Returns:
        Merged and sorted results with rrf_score
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}
    
    # Process BM25 results
    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.get("id", "")
        if doc_id:
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            doc_map[doc_id] = doc
    
    # Process vector results
    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc.get("id", "")
        if doc_id:
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            if doc_id not in doc_map:
                doc_map[doc_id] = doc
    
    # Merge and sort by RRF score
    merged_results = []
    for doc_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = rrf_score
        merged_results.append(doc)
    
    logger.info(f"RRF fusion: {len(bm25_results)} BM25 + {len(vector_results)} vector → {len(merged_results)} merged")
    
    return merged_results


def hybrid_search(
    query: str,
    bm25_index,
    vector_store,
    top_k: int = 10,
    fetch_k: int = 20,
    rrf_k: int = 60
) -> List[Dict[str, Any]]:
    """
    Perform hybrid search combining BM25 and vector retrieval.
    
    Args:
        query: Search query
        bm25_index: BM25Index instance
        vector_store: Vector store instance with search method
        top_k: Final number of results to return
        fetch_k: Number of results to fetch from each retriever
        rrf_k: RRF constant for fusion
    
    Returns:
        Merged results sorted by RRF score
    """
    # Get BM25 results
    bm25_results = []
    if bm25_index and bm25_index.is_available():
        bm25_results = bm25_index.search(query, top_k=fetch_k)
    
    # Get vector results
    vector_results = []
    if vector_store:
        vector_results = vector_store.search(query, top_k=fetch_k)
    
    # Merge with RRF
    merged_results = reciprocal_rank_fusion(bm25_results, vector_results, k=rrf_k)
    
    # Return top-k
    return merged_results[:top_k]
