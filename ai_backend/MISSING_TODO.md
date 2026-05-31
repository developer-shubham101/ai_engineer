# P0 — Must fix (correctness bugs)

- [x] **BM25 not refreshed after document add/update** — `app/modules/core/document_manager.py`
    - `_bm25_dirty = True` now set in `add_document_to_rag_local()` after every successful vector store write
    - `_refresh_bm25_if_dirty()` called at end of `update_document_version()`
    - Directory seed path now calls `_build_bm25_index()` + resets flag after completion
    - `retrieve_documents()` in `rag_orchestrator.py` flushes dirty flag before search

- [x] **Chroma metadata filter ignored in search** — `app/modules/vector_db/chroma_impl.py`
    - `search_documents()` passes `where=metadata_filter` to `query_collection()` which forwards it to `collection.query()`

- [x] **FAISS deletion leaves stale vectors in index** — `app/modules/vector_db/faiss_vector_store.py`
    - `_compact()` method rebuilds index from `self.documents.values()`
    - Called in `delete_document()` and `delete_ids()`
    - `get_collection_info()` returns `len(self.documents)` not `index.ntotal`

- [x] **RBAC filter in retrieve_documents() is too permissive** — `app/modules/llm/rag_orchestrator.py`
    - Level-based check: `ROLE_LEVELS[user_role] >= SENSITIVITY_LEVELS[sensitivity]`
    - `department_confidential` requires both level AND same dept
    - `personal` requires owner match OR HR+ level

- [x] **Context truncated to 500 chars per document** — `app/modules/llm/rag_orchestrator.py`
    - `build_context()` uses `[:2000]` per document
    - `_build_messages()` uses `[:2000]` per document (no duplicate truncation)

# P1 — High impact, low risk

- [x] **Query expansion disabled** — `app/modules/llm/rag_orchestrator.py`
    - `use_expansion=True` in `process_query()` call

- [x] **Query variants computed but discarded** — `app/modules/llm/rag_orchestrator.py`
    - Multi-variant retrieval: top-3 unique variants run in parallel across BM25 + vector
    - Results fused with additional RRF pass before RBAC filtering

- [x] **Reranker instantiated on every query** — `app/modules/llm/rag_orchestrator.py`
    - `self._reranker` lazy singleton on `RAGOrchestrator`; instantiated once, reused

- [x] **FAISS save path not guaranteed to exist** — `app/modules/vector_db/faiss_vector_store.py`
    - `_save_index()` calls `Path(self.file_path).parent.mkdir(parents=True, exist_ok=True)`

- [x] **Dead hybrid_search() helper uses wrong interface** — `app/modules/vector_db/hybrid_retrieval.py`
    - Dead function removed; only `reciprocal_rank_fusion()` remains

# P2 — Quality improvements

- [x] **RRF weights are static and query-type-unaware** — `app/modules/vector_db/hybrid_retrieval.py`
    - `reciprocal_rank_fusion()` accepts `bm25_weight` / `vector_weight` params
    - `retrieve_documents()` passes weights based on `processed_query.query_type`
    - POLICY/FACTUAL → `bm25_weight=1.5, vector_weight=1.0`; else reversed

- [x] **normalize_query() strips apostrophes before spell correction** — `app/modules/vector_db/query_preprocessor.py`
    - `process_query()` runs `correct_spelling()` on original query before `normalize_query()`

- [x] **FAISS uses L2 distance instead of cosine** — `app/modules/vector_db/faiss_vector_store.py`
    - Uses `IndexFlatIP` with `faiss.normalize_L2()` before `index.add()` and `index.search()`

# P3 — Ops hardening

- [x] **BM25 tokenization too naive for enterprise identifiers** — `app/modules/vector_db/bm25_index.py`
    - `_tokenize()` uses `re.findall(r'[a-z0-9]+', text.lower())` — splits on hyphens, underscores, separators

- [x] **No offline retrieval benchmark harness** — `scripts/benchmark_retrieval.py`
    - Created: tracks Recall@3, MRR, latency p50/p95 against QA pairs from company documents
    - Run: `python scripts/benchmark_retrieval.py`
