"""Offline retrieval benchmark: Recall@3, MRR, latency p50/p95.

Measures the full pipeline: BM25 + vector hybrid, RRF fusion, RBAC filter, reranker.
Run: python scripts/benchmark_retrieval.py
"""
from __future__ import annotations

import asyncio
import statistics
import sys
import os
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# QA pairs derived from data/company documents
QA_PAIRS: List[Dict[str, Any]] = [
    {"question": "What is the PTO policy?", "relevant_keywords": ["pto", "paid time off", "vacation", "leave"]},
    {"question": "How do I request work from home?", "relevant_keywords": ["wfh", "work from home", "remote"]},
    {"question": "What are the RBAC roles?", "relevant_keywords": ["rbac", "role", "superadmin", "manager", "employee"]},
    {"question": "What is the password policy?", "relevant_keywords": ["password", "security", "authentication"]},
    {"question": "How do I submit an expense report?", "relevant_keywords": ["expense", "reimbursement", "report"]},
    {"question": "What is the onboarding process?", "relevant_keywords": ["onboarding", "new hire", "orientation"]},
    {"question": "What are the leave policies?", "relevant_keywords": ["leave", "sick", "maternity", "paternity"]},
    {"question": "How do I access the VPN?", "relevant_keywords": ["vpn", "remote access", "network"]},
]

# SuperAdmin user — passes all RBAC filters so we measure retrieval quality, not access control
_SUPERADMIN_USER = {"role": "SuperAdmin", "department": "General", "user_id": "benchmark"}


def _is_relevant(doc: Dict[str, Any], keywords: List[str]) -> bool:
    text = (doc.get("text", "") + str(doc.get("metadata", {}))).lower()
    return any(kw in text for kw in keywords)


async def run_benchmark(top_k: int = 3) -> None:
    from app.modules.integration import get_container
    container = get_container()
    container.initialize()

    orchestrator = container.get_rag_orchestrator()

    recall_scores: List[float] = []
    mrr_scores: List[float] = []
    latencies: List[float] = []

    print(f"\n{'='*60}")
    print(f"Retrieval Benchmark  top_k={top_k}  queries={len(QA_PAIRS)}")
    print(f"Pipeline: BM25 + vector hybrid → RRF → RBAC → reranker")
    print(f"{'='*60}")

    for qa in QA_PAIRS:
        question = qa["question"]
        keywords = qa["relevant_keywords"]

        t0 = time.perf_counter()
        results = await orchestrator.retrieve_documents(
            query=question, user=_SUPERADMIN_USER, top_k=top_k
        )
        latency_ms = (time.perf_counter() - t0) * 1000
        latencies.append(latency_ms)

        hits = [_is_relevant(d, keywords) for d in results]
        recall = 1.0 if any(hits) else 0.0
        recall_scores.append(recall)

        mrr = 0.0
        for rank, hit in enumerate(hits, start=1):
            if hit:
                mrr = 1.0 / rank
                break
        mrr_scores.append(mrr)

        status = "✓" if recall else "✗"
        print(f"  {status} [{latency_ms:6.1f}ms] {question[:55]}")

    latencies_sorted = sorted(latencies)
    n = len(latencies_sorted)
    p50 = statistics.median(latencies_sorted)
    p95 = latencies_sorted[min(int(n * 0.95), n - 1)]

    print(f"\n{'='*60}")
    print(f"  Recall@{top_k}   : {sum(recall_scores)/len(recall_scores):.3f}")
    print(f"  MRR        : {sum(mrr_scores)/len(mrr_scores):.3f}")
    print(f"  Latency p50: {p50:.1f}ms   p95: {p95:.1f}ms")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(run_benchmark())
