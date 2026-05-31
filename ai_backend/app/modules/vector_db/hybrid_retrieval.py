"""Hybrid retrieval combining BM25 and vector search with RRF fusion."""
from __future__ import annotations

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    bm25_results: List[Dict[str, Any]],
    vector_results: List[Dict[str, Any]],
    k: int = 60,
    bm25_weight: float = 1.0,
    vector_weight: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Merge BM25 and vector results using weighted Reciprocal Rank Fusion.

    RRF formula: score = bm25_weight * 1/(k+rank_bm25) + vector_weight * 1/(k+rank_vector)

    Args:
        bm25_results: Results from BM25 search with 'id' field
        vector_results: Results from vector search with 'id' field
        k: RRF constant (default 60, standard value)
        bm25_weight: Weight applied to BM25 rank scores (higher = favour keyword matches)
        vector_weight: Weight applied to vector rank scores (higher = favour semantic matches)

    Returns:
        Merged and sorted results with rrf_score
    """
    rrf_scores: Dict[str, float] = {}
    doc_map: Dict[str, Dict[str, Any]] = {}

    for rank, doc in enumerate(bm25_results, start=1):
        doc_id = doc.get("id", "")
        if doc_id:
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + bm25_weight / (k + rank)
            doc_map[doc_id] = doc

    for rank, doc in enumerate(vector_results, start=1):
        doc_id = doc.get("id", "")
        if doc_id:
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + vector_weight / (k + rank)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

    merged_results = []
    for doc_id, rrf_score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        doc = doc_map[doc_id].copy()
        doc["rrf_score"] = rrf_score
        merged_results.append(doc)

    logger.info(
        "RRF fusion: %d BM25 (w=%.1f) + %d vector (w=%.1f) → %d merged",
        len(bm25_results), bm25_weight, len(vector_results), vector_weight, len(merged_results),
    )
    return merged_results
