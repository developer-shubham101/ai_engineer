# TODO — Vector DB, Retrieval Pipeline & RAG Quality

All items verified against source code. ✅ = implemented and confirmed

## P0 — Correctness bugs

- [x] **BM25 not refreshed after document add/update** — `app/modules/core/document_manager.py` ✅
    - `_bm25_dirty = True` set after every `add_document_to_rag_local()` write
    - `_refresh_bm25_if_dirty()` called at end of `update_document_version()`
    - Directory seed path calls `_build_bm25_index()` + resets flag
    - `retrieve_documents()` flushes dirty flag before search

- [x] **Chroma metadata filter ignored in search** — `app/modules/vector_db/chroma_impl.py` ✅
    - `search_documents()` passes `where=metadata_filter` to `query_collection()`

- [x] **FAISS deletion leaves stale vectors** — `app/modules/vector_db/faiss_vector_store.py` ✅
    - `_compact()` iterates `self.documents.items()` directly (sparse key bug fixed)
    - Called in `delete_document()` and `delete_ids()`
    - `get_collection_info()` returns `len(self.documents)`

- [x] **RBAC filter too permissive** — `app/modules/llm/rag_orchestrator.py` ✅
    - Level-based check: `ROLE_LEVELS[user_role] >= SENSITIVITY_LEVELS[sensitivity]`
    - `department_confidential` requires both level AND same dept

- [x] **Context truncated to 500 chars** — `app/modules/llm/rag_orchestrator.py` ✅
    - `build_context()` and `_build_messages()` both use `[:2000]`

## P1 — Quality improvements

- [x] **Query expansion disabled** ✅ — `use_expansion=True`
- [x] **Query variants discarded** ✅ — top-3 variants used across BM25 + vector, fused with RRF
- [x] **Reranker instantiated on every query** ✅ — `self._reranker` lazy singleton
- [x] **QueryPreprocessor instantiated on every query** ✅ — `self._preprocessor` singleton in `__init__`
- [x] **FAISS save path not guaranteed to exist** ✅ — `mkdir(parents=True, exist_ok=True)`
- [x] **Dead `hybrid_search()` helper** ✅ — removed
- [x] **RRF weights static** ✅ — `bm25_weight`/`vector_weight` params, driven by `query_type`
- [x] **Spell correction ran after normalization** ✅ — correction runs on original before normalize
- [x] **BM25 tokenization whitespace-only** ✅ — `re.findall(r'[a-z0-9]+', ...)`
- [x] **FAISS used L2 instead of cosine** ✅ — `IndexFlatIP` + `normalize_L2`
- [x] **provider_factory.py missing GPT/HF plugins** ✅
    - `OpenAIProviderPlugin` registered for `openai`
    - `HuggingFaceProviderPlugin` registered for `huggingface`
    - `gpt` → `openai` and `hf` → `huggingface` aliases resolved in `create_provider()`

## P2 — Ops hardening

- [x] **No offline retrieval benchmark** ✅ — `scripts/benchmark_retrieval.py`
    - Uses `rag_orchestrator.retrieve_documents()` — full pipeline (BM25 + vector + RRF + RBAC + reranker)
    - Tracks Recall@3, MRR, latency p50/p95

## Test coverage

- [x] `test_faiss_vector_store.py` — `test_delete_and_compact`, `test_compact_preserves_remaining_docs` ✅
- [x] `test_bm25_hybrid.py` — `test_bm25_tokenizer_splits_identifiers`, `test_rrf_weighted_fusion`, `test_bm25_freshness_after_api_add` ✅
- [x] `test_rag_orchestrator.py` — `test_rbac_level_filtering`, `test_context_length` ✅

## Definition of done — all complete ✅

- [x] BM25 fresh after every add/update/seed path
- [x] Chroma metadata filter applied
- [x] FAISS `_compact()` correct sparse-key mapping
- [x] FAISS `document_count` reflects active documents
- [x] `hybrid_search()` helper deleted
- [x] Singleton reranker and preprocessor — no per-query re-init
- [x] Query expansion enabled; all variants used
- [x] RBAC level-based check
- [x] Context 2000 chars per document
- [x] Weighted RRF — query-type-driven
- [x] Spell correction before normalization
- [x] BM25 tokenizer handles enterprise identifiers
- [x] FAISS cosine similarity
- [x] All providers registered (local, google, openai/gpt, huggingface/hf, colabllm, llamaserver)
- [x] Benchmark measures full pipeline
- [x] Delete+compact, BM25 freshness, RBAC, context-length tests added

---

# Query Input Robustness — Misspell / Bad Grammar / No Spaces

All items implemented in `app/modules/vector_db/query_preprocessor.py` unless noted. All complete ✅

## P0 — Active bugs (fixed)

- [x] **Domain words mis-corrected by spell checker** — `query_preprocessor.py` ✅
    - `pyspellchecker` mis-corrected domain terms: `rbac` → `race`, `pto` → `pro`, `wfh` → `who`
    - Fixed: domain vocabulary (all expansion keys + `rbac`, `pto`, `wfh`, `ooo`, `rto`, `sso`, `mfa`, `onboarding`, `offboarding`, `payroll`, `reimbursement`) loaded into `spell_checker.word_frequency` at init

- [x] **Apostrophe stripped before spell correction** — `query_preprocessor.py` ✅
    - `normalize_query()` regex now preserves apostrophes: `[^a-z0-9\s\-']`
    - `"what's"` no longer becomes `"what s"` before spell correction

## P1 — Missing capabilities (implemented)

- [x] **Words run together without spaces** — `query_preprocessor.py` ✅
    - New `split_concatenated_words()` uses `wordninja` to split `"leavepolicy"` → `"leave policy"`
    - Applies to words > 8 chars that are all-alpha; called before spell correction in `process_query()`
    - Dependency added: `wordninja` in `requirements.txt`

- [x] **Repeated characters** — `query_preprocessor.py` ✅
    - New `remove_repeated_chars()` collapses 3+ repeated chars to 2: `"leeeeave"` → `"leeave"`
    - Called first in `process_query()` before all other steps; zero new dependencies

- [x] **Digit-word run-together** — `query_preprocessor.py` ✅
    - `correct_spelling()` now splits digit-word combos before the digit-skip guard
    - `"3day"` → `["3", "day"]`, `"2weeks"` → `["2", "weeks"]`; BM25 tokenizer gets proper tokens

- [x] **Grammar issues not handled** — `query_preprocessor.py` ✅
    - New `_looks_broken()` returns `True` when unknown-word ratio > 30% (skips queries ≤ 3 words)
    - `process_query()` auto-enables `use_llm_rewrite=True` when broken query detected and `llm_provider` is available
    - `rewrite_with_llm()` was already implemented; now fires automatically on broken input

## P2 — Retrieval tuning (implemented)

- [x] **Vague single/two-word queries get noisy candidates** — `rag_orchestrator.py` ✅
    - `retrieve_documents()` detects queries ≤ 2 words and sets `retrieval_k = max(top_k * 6, 30)`
    - Normal queries remain at `max(top_k * 4, 20)`

## `process_query()` pipeline order (final)

1. `remove_repeated_chars()` — collapse `leeeeave` → `leeave`
2. `split_concatenated_words()` — split `leavepolicy` → `leave policy` (wordninja)
3. `correct_spelling()` — spell correction with digit-word split + domain word protection
4. `normalize_query()` — lowercase, strip special chars (apostrophes preserved)
5. `expand_query()` — acronym/synonym expansion
6. `_looks_broken()` + auto LLM rewrite — grammar repair if provider available

## New dependency

```text
wordninja  # word segmentation for run-together words (added to requirements.txt)
```
