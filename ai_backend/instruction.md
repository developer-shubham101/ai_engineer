---
Tooling Guidance

- High-priority files for context-aware tooling:
  - `docs/Role-based enterprise AI design.md` (role definitions, sensitivity tiers).
  - `docs/requirment.md` (system goals, stack decisions, current backlog).
  - `app/services/rag_local_service.py`, `app/api_routes_local.py`, `scripts/seed_examples.py` (core RAG/RBAC logic).
  - `instruction.txt` (this file) for latest progress + tasks.
- Secondary references: `docs/task_for_ai.md`, `docs/how to use properly cloud and local model.md`.
- Skip/ignore during automated scans:
  - `app/chroma_storage/`, `app/data/sessions/`, `models/` (heavy binaries/db files).
  - `data/missions_output/` unless specifically working on dataset curation.
  - `.idea/`, `.cache/`, `logs/`.
- Use `scripts/local_tree_printer.py` if the current tree needs refreshing; update `LOCAL_REPO_PATH` before running.
---
Requirements Snapshot (Role-Based Enterprise AI)

- Roles: Employee, Manager/Lead, HR, Finance, IT Support, Legal/Compliance, Executive (per `docs/Role-based enterprise AI design.md`).
- Knowledge domains: HR, Finance, IT, Engineering, Sales, Legal, General policy, Confidential datasets.
- Sensitivity tiers:
  - `public_internal`: all staff
  - `department_confidential`: department scoped
  - `role_confidential`: HR/Managers
  - `highly_confidential`: Legal/Execs
  - `personal`: user-specific records
- Behavioral guardrails: deny restricted answers with summaries, provide manager workflows, legal guidance, onboarding steps, and compliance guardrails that block disallowed requests.
- Business features to support: role-based answers, workflow guidance, policy interpretation, multi-department chat support, decision support, document generation, compliance guardrails.

Implementation Status (2025-11-19)

- ✅ Local CPU-only RAG stack using ChromaDB + MiniLM embeddings + LlamaCpp per `docs/requirment.md`.
- ✅ RBAC metadata already defined (`public_summary`, `filtered_details`, `filtered_out_count` shaped responses).
- ✅ `app/services/rag_local_service.py` implements ingestion, query filtering, and local inference; `app/api_routes_local.py` exposes ingestion/query endpoints.
- ✅ `scripts/seed_examples.py` seeds HR/Finance docs with metadata for RBAC validation.
- ✅ Support chat foundation deployed: SQLite session store + lifecycle endpoints + `/api/local/query` session awareness (history-aware prompts, message logging).
- ⚠️ Compliance monitoring/KPIs/audit hooks unspecified.

Support Chat Implementation (phase 1 complete)

- Storage: lightweight SQLite DB (`app/data/support_sessions.db`) accessed via `app/services/support_chat.py`. Tables:
  - `sessions(id TEXT PRIMARY KEY, role TEXT, department TEXT, created_at, updated_at)`.
  - `messages(id INTEGER PK, session_id TEXT FK, speaker TEXT, content TEXT, created_at)`.
- Session ID handling:
  - Clients pass `X-Session-Id` header on every chat request.
  - `POST /api/local/session/start` creates new session (body captures name/sex/position/category/notes). Profile info is stored as a “system” message rather than dedicated columns.
  - `POST /api/local/session/end` deletes that session + any logged history. App restart wipes previous sessions because DB resets on startup.
- Chat endpoint reuse:
  - `POST /api/local/query` now accepts optional header `X-Session-Id` and optional request body field `category`.
  - When a session header is present, the handler:
    - Verifies session existence, fetches last 5 turns, builds an enriched `llm_prompt_prefix` via `support_chat.build_prompt_prefix`, and logs every user/assistant turn.
    - Stores profile + history solely through the `messages` log, satisfying the “no extra columns” requirement.
  - Without a session header the behavior matches the original single-turn query flow.
- History scope:
  - Store full turn history in SQLite; when building LLM prompt, include the last N turns (configurable, default 5) plus session profile so model “remembers” name/role/position.
  - Keep short contexts during testing but allow admins to configure max turns.
- Dependencies to add:
-  - New module `app/services/support_chat.py` wrapping SQLite access + history assembly. Provides helpers: `create_session`, `touch_session`, `end_session`, `store_message`, `fetch_recent_messages`, and `build_prompt_prefix`.
-  - FastAPI router enhancements in `app/api_routes_local.py` for `/api/local/session/start`, `/api/local/session/end`, and session-aware `/api/local/query`.
- Category-to-prompt mapping:
  - Static dict (e.g., `support_categories = {"HR": {...}, "IT": {...}}`) specifying allowed departments/metadata hints and prompt templates (prefill instructions like “You are an HR support assistant…”).
- Dependencies to add:
  - New module `app/services/support_chat.py` wrapping SQLite access + history assembly.
  - FastAPI router `app/api_routes_support.py` registering `/api/support/*` endpoints (imported in main app).

Service Review — `app/services/rag_local_service.py`

- Handles local embeddings (SentenceTransformer `all-MiniLM-L6-v2`) with optional on-disk cache under `embeddings_models/`; lazy-loads LlamaCpp models from `models/`.
- `add_document_to_rag_local` chunks text (512/64 overlap), sanitizes metadata (`sensitivity`, `department`, `allowed_roles`, `owner_id`, `public_summary` etc.), computes embeddings, and writes to Chroma via `chroma_utils`.
- `query_local_rag` embeds the query, hits Chroma, and applies RBAC filtering with helper `_allowed_by_metadata`:
  - `personal` limited to owner or HR/Legal/Executive.
  - `highly_confidential` restricted to Legal/Executive.
  - `role_confidential` honors `allowed_roles`, defaulting to HR/Legal/Executive fallback.
  - `department_confidential` checks department match or HR/Legal/Executive.
  - `public_internal` default allow.
  - Collects `public_summaries`, `filtered_details`, `filtered_out_count` for UX.
- Optional LLM invocation constructs prompt from visible context only; fails fast if no local GGUF found unless `llm_instance` injected via `initialize_local_rag`.
- Utility helpers: `_sanitize_metadata_dict`, `_chunk_text_basic`, `seed_from_file`, `update_metadata`, `clear_collection`. No streaming responses or batching yet; consider caching requester-role policies if perf becomes issue.

API Endpoints — `app/api_routes_local.py`

- `POST /api/local/query`
  - Body: `QueryRequest {question, top_k, use_llm, max_tokens, category?}`.
  - Depends on `get_requester()` (API key header `X-API-Key`) for role/department context; optional `X-Session-Id` header activates chat mode.
  - Calls `query_local_rag`; returns `QueryResponse {answer, retrieved[], context}` with RBAC-enforced docs.
  - Detailed behavior:
    - Headers: optional `X-API-Key` resolving to `{user_id, role, department}` via `app.services.auth.get_user_from_api_key`; defaults to Guest if missing. Optional `X-Session-Id` links to SQLite-backed history.
    - Logging: records requester role/user plus question string for traceability.
    - Session-aware flow:
      - Validates session id, touches timestamps, loads last 5 turns via `support_chat.fetch_recent_messages`, and injects them into `llm_prompt_prefix` (category influences persona instructions).
      - After answering, appends both the user question and assistant reply to the `messages` table so the model “remembers” details in follow-up turns.
    - Response shaping:
      - Collates `documents/metadatas/ids/distances` from service into `RetrievedDoc` objects, flattening nested lists if needed.
      - `answer` comes from LLM output when `use_llm=True`; otherwise friendly default (“Review retrieved items” or “No relevant documents”).
      - `context` echoes the concatenated visible chunks for downstream clients.
    - RBAC enforcement occurs inside service; the endpoint never exposes filtered documents but does return `retrieved` metadata so clients can explain access decisions.
- `POST /api/local/add`
  - Body: `AddDocRequest {source_name, text, metadata}`; auto-fills `department`, `sensitivity`, `ingested_by`.
  - Validates `sensitivity` against allowed set before calling `add_document_to_rag_local`.
  - Detailed behavior:
    - Caller supplies raw text; endpoint chunks on the service side, embeds locally, and stores with metadata so future `/query` calls can retrieve the material for qualified roles.
    - Metadata guardrails ensure every doc carries `department` + `sensitivity`; requester identity is stamped via `ingested_by` for audit.
    - Returns `{message, chunk_count}` so clients know how many vector chunks are now eligible for retrieval in the RAG index.
- `POST /api/local/add-file`
  - Multipart upload (text files ≤5 MB) plus query params `department`, `sensitivity`.
  - Decodes UTF-8, builds metadata, invokes `add_document_to_rag_local`.
  - Detailed behavior:
    - Accepts `UploadFile` plus optional query parameters; automatically rejects empty or >5 MB payloads to protect local CPU deployments.
    - Reads bytes, decodes to text, tags metadata with requester role via `ingested_by`, then sends to the same ingestion pipeline as `/add`.
    - Response mirrors `/add`, confirming chunk counts so operators know when the knowledge base now contains new RBAC-tagged content for downstream queries.
- `POST /api/local/seed`
  - No body; triggers `seed_from_file()` to ingest default `data/company_overview.txt` contents.
  - Detailed behavior:
    - Convenience endpoint for bootstrapping demos: it loads the canonical company overview and pushes it through the chunk/embedding workflow without manual uploads.
    - Returns a chunk count so users know what content the `/query` endpoint can now answer questions about.
- `POST /api/local/clear`
  - Requires requester role `Executive` or `Legal`; otherwise 403.
  - Calls `clear_collection()` to wipe Chroma for demos.
  - Detailed behavior:
    - Ensures only high-trust roles can flush the knowledge base; other roles receive a clear denial message.
    - Internally invokes `chroma_utils.delete_all_documents`, removing every chunk so future queries produce “no data” responses until reseeded.
- `POST /api/local/session/start`
  - Body: `SupportSessionStartRequest {session_id?, name?, sex?, position?, category?, notes?}`; returns `SupportSessionStartResponse {session_id, message}`.
  - Creates a new SQLite session row (role/department captured from requester) and logs a “system” message summarizing any provided profile details so future turns can reference them.
  - Clients should store the returned `session_id` and pass it via `X-Session-Id` header on every `/api/local/query`.
- `POST /api/local/session/end`
  - Body: `SupportSessionEndRequest {session_id}`; returns `SupportSessionEndResponse`.
  - Deletes the session + all messages. Queries referencing the ended session id will receive a 404 and must start a new session.
- Shared dependencies: `get_requester` (simple API-key auth) and `validate_metadata`. Extend here when adding support-chat endpoints or new workflows.

Progress Log — 2025-11-19

Completed today:
- Reviewed repo structure (`instruction.txt`).
- Extracted requirements and behavior expectations from `docs/Role-based enterprise AI design.md`.
- Captured current backend capabilities from `docs/requirment.md`.

Pending / Next Tasks:
- Produce RBAC matrix mapping each role to allowed categories + sensitivity tiers, including fallback summary behavior.
- Define request-time enforcement flow (inputs: role, department, sensitivity, personal ownership) and failure messaging.
- Outline how ingestion tags documents/chunks with the sensitivity + department metadata (instructions for future scripts).
- Design support-focused chat API: session schema, category prompts, storage strategy, escalation hooks.
- Document compliance plan: metrics, audit log format, alerting for blocked queries, periodic RBAC tests.