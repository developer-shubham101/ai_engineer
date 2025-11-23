# AI Engineering Project Roadmap

## PHASE 1 — Stability, Accuracy & Core Intelligence

### 1. Increase LLM Context + Token-Safe Pipeline (Completed)

* Increase `n_ctx` to avoid overflow
* Add retry wrapper `_call_llm_with_retry()`

### 2. Add Local Reranker

* Use cross-encoder or BGE reranker
* Pipeline: Chroma → Top-K → Reranker → Top-N → LLM

### 3. Add Prompt Caching

* Cache embeddings, chunks, prefixes, and responses

### 4. Add Explainability Endpoint `/rag/explain`

* Return retrieved, filtered, selected chunks and scores

---

## PHASE 2 — Enterprise-Ready Assistants

### 5. Intelligent Profile Routing

* Based on role: HR, IT, Finance, Legal, Manager

### 6. Onboarding Memory Expansion

* Store preferences, writing style, past troubleshooting

### 7. Query Reformulation

* Rewrite queries to boost retrieval accuracy

---

## PHASE 3 — RAG Engine Upgrades

### 8. Hybrid Retrieval

* Combine BM25 + Vector + Reranker

### 9. Document Versioning

* Track active/archived versions

### 10. Chunk Summaries

* Auto-summarize chunks during ingestion

---

## PHASE 4 — Security & Governance

### 11. Redaction Mode

* Replace sensitive parts with `[[REDACTED]]`

### 12. Full Audit Logging

* Log all retrievals, denied chunks, and decisions

---

## PHASE 5 — Analytics & Dashboards

### 13. Usage Analytics Dashboard

* Track top questions, docs, errors, latency

### 14. Department Knowledge Health

* Coverage reports for HR, IT, Finance, Legal

---

## PHASE 6 — Assistant Intelligence

### 15. Autonomous Routing

* Decide between RAG, LLM, follow-up question, or ticket

### 16. Lightweight Reasoning Models

* Use 2B–3B models for summarization, classification

### 17. Ticketing Workflow

* Auto create and summarize support tickets

---

## PHASE 7 — Offline Fine-Tuning / LoRA

### 18. LoRA Fine-Tuning

* Train adapters on internal policies and workflows

---

## PHASE 8 — Developer Experience & Scaling

### 19. Background Ingestion Queue

* Use FastAPI BackgroundTask or Celery

### 20. Multi-Model Support

* Switch between models dynamically

### 21. Environment-Based Config Loader

* Dev vs Production profiles

---

## TOP 10 NEXT STEPS

1. Reranker
2. Prompt caching
3. `/rag/explain`
4. Assistant routing
5. Query reformulation
6. Hybrid retrieval
7. Document versioning
8. Redaction rules
9. Analytics dashboard
10. Autonomous routing logic
