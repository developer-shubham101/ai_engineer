**Single Source of Truth for Project Context**

> **Instructions for Copilot**: When generating code, prioritize the information in this file. Use only this file + the currently open buffer for context. Do not read the entire repository unless explicitly asked.

---

## 1. Project Summary

A **versatile enterprise RAG (Retrieval Augmented Generation) system** designed for learning and offline testing, now extending beyond local-only capabilities. The system simulates an enterprise AI assistant for a fictional company with role-based access control (RBAC), document retrieval, sentiment analysis, and multi-turn chat sessions. Initially a fully local, CPU-only system, it now also integrates with **Google Gemini API**, with plans to support other external APIs (e.g., Hugging Face) in the future. It currently utilizes local components like Mistral-7B-Instruct (GGUF), MiniLM embeddings, ChromaDB, and FastAPI.

---

## 2. High-Level Architecture

```
User Request → FastAPI → RAG Pipeline → RBAC Filter → LLM (Local or Google) → Response
                                    ↓
                            Chroma Vector DB
                                    ↓
                        Local Embeddings (MiniLM)
```

### Main Components

- **Backend**: FastAPI application (`app/main.py`, `app/api_routes_rag.py`, `app/api_routes_auth.py`)
- **Configuration**: Centralized settings in `app/config.py` (includes JWT configuration)
- **Dependency Injection**: `app/dependencies.py` - Provides service dependencies and authentication for testability
- **Authentication & Authorization**:
  - `user_service.py` - User management with SQLite database, password hashing (bcrypt), **user profile storage (user_meta table)**
  - `auth.py` - JWT token generation and verification
  - `api_routes_auth.py` - Authentication endpoints (`/api/auth/token`)
  - Bearer token authentication via `Authorization` header
  - Role-based access control (RBAC) for endpoints
  - **Auto-session creation on login using user_id**
- **RAG Services**: 
  - `rag_local_service.py` - Core RAG with RBAC and data indexing logic for local models (modular architecture).
  - `google_models.py` - RAG logic for Google's generative models.
  - `model_manager.py` - Handles loading and caching of local LLM instances.
  - `prompt_builder.py` - Constructs prompts and manages token budgets.
- **Vector DB**: ChromaDB (persistent storage in `app/chroma_storage/`)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2) - loaded from `embeddings_models/`
- **Local LLM**: Mistral-7B-Instruct-v0.2.Q3_K_M.gguf via llama-cpp-python (default model, with optional dynamic selection)
- **Support Chat**: SQLite-based session management (`app/data/support_sessions.db`) - **sessions auto-created on login using user_id**
- **Sentiment Analysis**: Local classifier using scikit-learn + sentence-transformers
- **Utilities**: Centralized paths, constants, shared functions (`app/services/utility.py`)

### Key Directories

- `app/` - Main application code
  - `services/` - Business logic modules (modular, testable functions)
    - `user_service.py` - User management with SQLite, **user_meta table for profiles**
    - `auth.py` - JWT token handling
  - `utils/` - Utility modules (e.g., `doc_parser.py`)
  - `config/` - Configuration files (e.g., `onboarding_fields.json`, `config.py`)
  - `data/` - SQLite databases (**users.db with user_meta table**, support_sessions.db), seed data
  - `chroma_storage/` - ChromaDB persistence
  - `dependencies.py` - Dependency injection providers (auth, RBAC, services)
  - `api_routes_auth.py` - Authentication API routes
- `data/` - Document examples, company overview, seed files
- `models/` - Local LLM model files (*.gguf, *.ggml, *.bin)
- `embeddings_models/` - Local embedding model cache
- `sentiment/` - Sentiment classifier artifacts
- `scripts/` - Utility scripts (seed, test, download)
- `docs/` - Documentation

---

## 3. Key APIs / Functions

### Main API Endpoints

**Authentication** (`/api/auth/*`):
- `POST /api/auth/token` - Login with username/password, returns JWT token. **Auto-creates session using user_id. Returns user profile from user_meta table.**
  - Request: `{"username": "string", "password": "string"}`
  - Response: `{"access_token": "jwt_token", "token_type": "bearer", "user": {..., "profile": {...}}}`

**RAG Services** (`/api/rag/*`):
- `POST /api/rag/{model_provider}/query` - RAG query with RBAC filtering, optional LLM generation, session-aware. **Public endpoint** (no auth required, Guest users see public docs only). **Uses user_id from JWT token as session identifier (no X-Session-Id header needed)**. `model_provider` can be `local` or `google`.
- `POST /api/rag/add` - Add document via JSON (text + metadata). **Creates version 1.0**. **Requires**: SuperAdmin, HR, Manager, or Employee role.
- `POST /api/rag/add-file` - Upload text file (≤5MB) for ingestion. **Creates version 1.0**. **Requires**: SuperAdmin, HR, Manager, or Employee role.
- `POST /api/rag/seed` - Seed default `data/companyData`. **Requires**: SuperAdmin role only.
- `POST /api/rag/clear` - Clear Chroma collection. **Requires**: SuperAdmin role only.
- `POST /api/rag/sentiment` - Analyze sentiment/tone of text. **Requires**: SuperAdmin role only.
- `GET /api/rag/sentiment/stats` - Get sentiment statistics. **Requires**: SuperAdmin role only.

**Document Versioning** (`/api/rag/documents/*`):
- `POST /api/rag/documents/update` - Update document (creates new version, non-destructive). **Requires**: SuperAdmin, HR, Manager, or Employee role.
  - Request: `{"document_id": "doc_abc...", "text": "...", "version_notes": "Fixed typos", "status": "published"}`
  - Response: `{"message": "...", "document_id": "...", "version": "2.0", "chunk_count": 5, "status": "published"}`
- `GET /api/rag/documents/list` - List all documents with filtering. Query params: `?department=HR&status=published&latest_only=true`. **Requires**: SuperAdmin, HR, Manager, or Employee role.
- `GET /api/rag/documents/{document_id}/versions` - Get version history for a document. **Requires**: SuperAdmin, HR, Manager, or Employee role.
- `GET /api/rag/documents/{document_id}/versions/{version}` - Get specific version with full content. **Requires**: SuperAdmin, HR, Manager, or Employee role.
- `GET /api/rag/documents/{document_id}/compare?version1=1.0&version2=2.0` - Compare two versions (shows diff). **Requires**: SuperAdmin, HR, Manager, or Employee role.
- `POST /api/rag/documents/{document_id}/archive` - Archive a version (soft-delete). **Requires**: SuperAdmin, HR, or Manager role.

### Core Service Functions

**RAG Services** (`rag_local_service.py`):
- `initialize_local_rag()` - Initialize Chroma client and collection.
- `add_document_to_rag_local(source_name, text, metadata, document_id, version, parent_version, status, version_notes, created_by)` - Asynchronously chunk, embed, store document **with versioning support**. Returns dict with `{ids, document_id, version, chunk_count}`.
- `update_document_version(document_id, text, metadata, version_notes, requester_id, status)` - Create new version of existing document (non-destructive).
- `get_document_version(document_id, version)` - Retrieve specific version with its chunks.
- `compare_document_versions(document_id, version1, version2)` - Compare two versions and return diff.
- `list_documents(department, status, latest_only)` - List all documents with filtering.
- `archive_document_version(document_id, version)` - Archive (soft-delete) a version.
- `query_local_rag(query_text, n_results, requester, llm_prompt_prefix, use_llm, max_tokens, session_id)` - Main orchestrator for RAG queries (calls modular sub-functions).
- `retrieve_documents(query_text, n_results)` - Handle embedding and ChromaDB retrieval.
- `normalize_chroma_result(result)` - Normalize ChromaDB response format.
- `filter_documents_by_rbac(raw_docs, raw_metadatas, raw_ids, raw_distances, requester)` - Apply RBAC filtering.
- `inject_tone_guidance(session_id, llm_prompt_prefix)` - Build tone-aware prompt prefix.
- `generate_rag_response(query_text, context_text, final_prefix, use_llm, max_tokens, session_id)` - Generate LLM response.
- `seed_from_file(force_reseed)` - Seed from file or directory.
- `clear_collection()` - Delete all documents.

**Version Tracking** (`version_tracking.py`):
- `init_version_db(reset_on_start)` - Initialize version tracking SQLite database.
- `create_version_record(document_id, version, source_name, chunk_ids, created_by, parent_version, status, version_notes, metadata)` - Store version metadata.
- `get_version_history(document_id)` - Get all versions of a document.
- `get_version(document_id, version)` - Get specific version metadata.
- `get_latest_version(document_id)` - Get most recent version.
- `update_version_status(document_id, version, status)` - Update version status (draft/published/archived).
- `get_documents_by_status(status)` - Filter documents by status.
- `list_all_documents(latest_only)` - List all documents.
- `generate_next_version(document_id)` - Calculate next semantic version number.

**Authentication & User Management** (`user_service.py`, `auth.py`):
- `init_user_db(reset_on_start)` - Initialize user database, **create user_meta table**, and seed dummy users with profiles.
- `authenticate_user(username, password)` - Authenticate user with credentials.
- `get_password_hash(password)` - Hash password using bcrypt.
- `verify_password(plain_password, hashed_password)` - Verify password.
- `create_access_token(user_data)` - Generate JWT token with user info.
- `verify_token(token)` - Verify and decode JWT token.
- **`get_user_meta(user_id, key)`** - Get single user profile field.
- **`get_all_user_meta(user_id)`** - Get all user profile fields as dict.
- **`set_user_meta(user_id, key, value)`** - Set/update user profile field.
- **`delete_user_meta(user_id, key)`** - Delete user profile field.

**Dependency Injection** (`dependencies.py`):
- `get_rag_service()` - Returns RAG service module for dependency injection in API routes.
- `get_current_user(credentials)` - Extract authenticated user from Bearer token (required auth).
- `get_current_user_optional(credentials)` - Optional authentication (returns None for Guest users).
- `require_roles(allowed_roles)` - Factory for role-based access control dependencies.

**Google RAG Service** (`google_models.py`):
- `query_google_rag(...)` - Asynchronously query using Google's generative models.

**Model Manager** (`model_manager.py`):
- `get_llm_instance(model_key)` - Lazy-load and cache LLM instances (default: mistral-7b-instruct-v0.2.Q3_K_M.gguf).
- `choose_model_for_task(task)` - Select model based on task type (only if `ENABLE_DYNAMIC_MODEL_SELECTION=True`).

**Prompt Builder** (`prompt_builder.py`):
- `build_prompt_with_selected_chunks(prefix, context_text, question)` - Constructs the final prompt for the LLM.
- `select_chunks_by_token_budget(...)` - Selects document chunks to fit within the model's context window.
- `_invoke_llm_with_chunk_budget(...)` - Asynchronously invokes the LLM with a token-budgeted prompt.

**Support Chat** (`support_chat.py`):
- `create_session(session_id, role, department)` - Create new session.
- `store_message(session_id, speaker, content)` - Store message with sentiment analysis.
- `fetch_recent_messages(session_id, limit)` - Get conversation history.
- `get_full_profile(session_id)` - Get user profile data.
- `get_next_missing_profile_key(session_id)` - Get next onboarding question.
- `set_profile_value(session_id, key, value)` - Set profile field.
- `build_prompt_prefix(requester, history, category)` - Build LLM prefix with context.

**Utilities** (`utility.py`):
- `get_embedding_model_instance()` - Singleton embedding model loader.
- `embed_texts(texts)` - Asynchronously embed list of texts.
- `chunk_text_basic(text, chunk_size, overlap)` - Text chunking.
- `sanitize_metadata_dict(meta)` - Sanitize metadata for Chroma.
- `normalize_tone_label(raw_tone)` - Map raw tone labels to canonical forms (angry, confused, happy, frustrated, polite, urgent, neutral).
- `get_local_embedding_model_path()` - Get embedding model path.
- `get_data_path(filename)`, `get_config_path(filename)` - Path helpers.

**Chroma Utils** (`chroma_utils.py`):
- `ensure_chroma_client(persist_directory, collection_name)` - Get/create Chroma client.
- `add_documents_to_collection(collection, documents, metadatas, ids, embeddings)` - Add docs.
- `query_collection(collection, query_embeddings, query_texts, n_results)` - Query vector DB.
- `delete_all_documents(collection, client, collection_name)` - Clear collection.

**Auth** (`auth.py`):
- `create_access_token(user_data, session_id)` - Generate JWT token with user info and optional session ID.
- `verify_token(token)` - Verify and decode JWT token.

**Sentiment** (`sentiment_classifier.py`):
- `get_global_sentiment()` - Get singleton classifier.
- `SentimentToneClassifier.predict_single(text)` - Predict sentiment and tone.

---

## 4. Coding Conventions

- **Language**: Python 3.10+ (Docker uses 3.11, local dev may be 3.10)
- **Type Hints**: Use type hints for function parameters and return types (`from __future__ import annotations`).
- **Formatting**: Follow PEP 8; use 4 spaces for indentation.
- **Imports**: 
  - Use absolute imports from `app.` namespace.
  - Group: stdlib, third-party, local.
  - Import from `utility.py` for shared paths/constants to avoid duplication.
- **Function Style**: 
  - Use descriptive names, docstrings for public functions.
  - Prefer composition over inheritance.
  - Use dependency injection (FastAPI `Depends`) for auth/requester.
- **Error Handling**: Use FastAPI `HTTPException` for API errors; log exceptions with context.
- **Logging**: Use module-level loggers (`logging.getLogger(__name__)`).
- **Constants**: Define in `app/config.py`; import rather than duplicate.
- **Circular Imports**: Avoid by importing shared constants from `app/config.py` or `app/services/utility.py`.

---

## 5. Important Files & Their Purpose

- `app/main.py` - FastAPI app entry point, lifespan handlers, endpoint registration.
- `app/api_routes_rag.py` - RAG API routes, request/response models, RBAC enforcement.
- `app/config.py` - Centralized configuration for the application.
- `app/services/rag_local_service.py` - Core RAG logic for local models: data indexing and retrieval.
- `app/services/google_models.py` - RAG logic for Google's generative models.
- `app/services/model_manager.py` - Handles loading and caching of local LLM instances.
- `app/services/prompt_builder.py` - Constructs prompts and manages token budgets.
- `app/services/support_chat.py` - SQLite session management, message storage, profile management, tone guidance.
- `app/services/utility.py` - **Centralized utilities**: paths, constants, embedding loader, text processing, metadata sanitization.
- `app/services/chroma_utils.py` - ChromaDB wrapper functions for client/collection operations.
- `app/services/auth.py` - API key to user mapping (role-based access).
- `app/services/sentiment_classifier.py` - Local sentiment/tone classification using scikit-learn.
- `app/logging_config.py` - Logging configuration.
- `app/config/onboarding_fields.json` - Onboarding question definitions.
- `requirements.txt` - Python dependencies (FastAPI, ChromaDB, sentence-transformers, llama-cpp-python, etc.).
- `Dockerfile` - Docker image definition (Python 3.11, port 5444).
- `docker-compose.yml` - Docker Compose configuration.

---

## 6. Data Models / Structures

### Request/Response Models

**QueryRequest**:
```python
{
  "question": str,
  "top_k": int = 3,
  "use_llm": bool = False,
  "max_tokens": int = 256,
  "category": Optional[str] = None
}
```

**QueryResponse**:
```python
{
  "answer": Optional[str],
  "retrieved": List[RetrievedDoc],
  "context": Optional[str]
}
```

**RetrievedDoc**:
```python
{
  "id": str,
  "text": str,
  "metadata": Optional[Dict[str, Any]],
  "distance": Optional[float]
}
```

**TokenRequest** (Login):
```python
{
  "username": str,
  "password": str
}
```

**TokenResponse** (Login):
```python
{
  "access_token": str,  # JWT token
  "token_type": str,    # "bearer"
  "user": {
    "user_id": str,
    "username": str,
    "role": str,
    "department": str,
    "profile": {  # User profile from user_meta table
      "name": str,
      "gender": str,
      "location": str,
      # ... other dynamic fields
    }
  }
}
```

**Requester (from Bearer token)**:
The user's identity is determined from the `Authorization: Bearer <token>` header.
```python
{
  "user_id": str,
  "username": Optional[str],
  "role": str,  # SuperAdmin, HR, Manager, Employee, Guest
  "department": str  # Engineering, Finance, HR, Legal, IT, Executive, etc.
}
```

**SupportSessionStartRequest**:
```python
{
    "session_id": Optional[str] = None,
    "name": Optional[str] = None,
    "sex": Optional[str] = None,
    "position": Optional[str] = None,
    "category": Optional[str] = None,
    "notes": Optional[str] = None
}
```

**Document Metadata**:
```python
{
  "source": str,
  "department": str,
  "sensitivity": str,  # public_internal, department_confidential, role_confidential, highly_confidential, personal
  "allowed_roles": Optional[List[str]],
  "owner_id": Optional[str],
  "public_summary": Optional[str],
  "ingested_at": str,
  "ingested_by": Optional[str]
}
```

**Session Message**:
```python
{
  "speaker": str,  # "user" or "assistant"
  "content": str,
  "created_at": str,
  "sentiment": Optional[str],
  "tone": Optional[str],
  "sentiment_meta": Optional[Dict]
}
```

**Sentiment Result**:
```python
{
  "ok": True,
  "result": {
    "text": str,
    "sentiment": str,  # positive, negative, neutral, unknown (on error)
    "tone": str,  # Canonical tones: angry, confused, happy, frustrated, polite, urgent, neutral
    "proba": {
      "sentiment": Dict[str, float],
      "tone": Dict[str, float]
    }
  }
}
```

**Model Selection Configuration** (`app/config.py`):
- `ENABLE_DYNAMIC_MODEL_SELECTION = False` - Flag to enable dynamic model selection based on task.
- `DEFAULT_MODEL_NAME = "mistral-7b-instruct-v0.2.Q3_K_M.gguf"` - Primary model to use.
- By default, system uses only the default model.
- If `ENABLE_DYNAMIC_MODEL_SELECTION=True` and default model not found, falls back to task-based selection:
  - `"small"` for summarization, classification, tagging, intent detection.
  - `"tiny"` for short chit-chat.
  - `"mistral"` for full RAG reasoning (default).

### RBAC Sensitivity Levels

1. `public_internal` - All authenticated users.
2. `department_confidential` - Same department or HR/Legal/Executive.
3. `role_confidential` - Specific roles (from `allowed_roles`) or HR/Legal/Executive.
4. `highly_confidential` - Legal/Executive only.
5. `personal` - Owner or HR/Legal/Executive.

---

## 7. Dev Notes

### Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Models** (if not present):
   - Embedding model: Run `scripts/download_embeddings_models.py` or place in `embeddings_models/all-MiniLM-L6-v2/`.
   - LLM model: Place `*.gguf` file in `models/` directory (e.g., `mistral-7b-instruct-v0.2.Q3_K_M.gguf`).

3. **Environment Variables** (optional):
   - `.env` file for optional cloud API keys (OpenAI, Google, HuggingFace) - not required for local-only mode.

### Running

**Development**:
```bash
uvicorn app.main:app --reload --port 5444
```

**Docker**:
```bash
docker-compose up --build
```

**Production**:
```bash
uvicorn app.main:app --host 0.0.0.0 --port 5444
```

### Testing

- No formal test suite currently.
- Use `api_test.md` for cURL examples.
- Use scripts in `scripts/` for manual testing:
  - `scripts/quick_test.py` - Quick API test.
  - `scripts/test_sentiment.py` - Sentiment classifier test.
  - `scripts/train_sentiment.py` - Train sentiment model.
  - `scripts/seed_examples.py` - Seed example documents.

### Key Behaviors

- **Lazy Loading**: Embedding model and LLM are loaded on first use (singleton pattern).
- **Model Selection**: By default, uses only `mistral-7b-instruct-v0.2.Q3_K_M.gguf`. Dynamic selection only if `ENABLE_DYNAMIC_MODEL_SELECTION=True` and default model not found.
- **Sentiment/Tone Detection**: Always returns canonical tone labels (angry, confused, happy, frustrated, polite, urgent, neutral). Never fails - defaults to "unknown"/"neutral" on error.
- **Auto-Seeding**: On startup, attempts to seed `data/companyData` if present.
- **Session Reset**: `support_chat.init_support_chat_db(reset_on_start=True)` resets DB on app start (change for production).
- **CORS**: Currently allows all origins (`allow_origins=["*"]`) - restrict in production.
- **File Upload Limit**: 5MB max for `/api/rag/add-file`.

### Directory Structure Requirements

- `app/chroma_storage/` - Created automatically by ChromaDB.
- `app/data/` - Created automatically for SQLite DBs.
- `models/` - Must contain `mistral-7b-instruct-v0.2.Q3_K_M.gguf` (or `.ggml`/`.bin` variant) for default operation.
  - If `ENABLE_DYNAMIC_MODEL_SELECTION=True`, can contain multiple models for task-based selection.
- `embeddings_models/all-MiniLM-L6-v2/` - Embedding model (auto-downloaded if missing).
- `sentiment/` - Created automatically for sentiment artifacts.

---

## 8. Instructions For Copilot

**When generating code:**

1. **Prioritize this file** - Use information from this file (`APP_CONTEXT.md`) as the primary source of truth.
2. **Use only this file + currently open buffer** - Do not read entire repository unless explicitly requested.
3. **Import from `app/config.py`** - Always import shared constants and configurations from `app/config.py`.
4. **Follow architecture patterns** - Maintain separation: routes in `api_routes_rag.py`, business logic in `services/`, utilities centralized. Use `model_manager.py` for LLM loading and `prompt_builder.py` for prompt construction.
5. **Respect RBAC** - Always enforce sensitivity levels when accessing documents.
6. **Use type hints** - Include type annotations for all function parameters and returns.
7. **Handle errors gracefully** - Use FastAPI `HTTPException` for API errors; log with context.
8. **Maintain singleton patterns** - Use `get_embedding_model_instance()` from `utility.py` for embeddings.
9. **Avoid circular imports** - Import shared constants from `app/config.py` or `app/services/utility.py`.
10. **Keep it local-first** - Prefer local solutions; cloud APIs are optional but supported.
11. **Model selection** - The endpoint path determines the model provider (`/api/rag/{model_provider}/query`).
12. **Tone normalization** - Always use `normalize_tone_label()` to ensure canonical tone labels (angry, confused, happy, frustrated, polite, urgent, neutral).
13. **Error handling for sentiment** - Never fail sentiment detection; always return defaults (sentiment="unknown", tone="neutral") on error.
14. **Dependency Injection** - Use `Depends(get_rag_service)` in API routes to inject the RAG service for better testability.
15. **Modular Functions** - When adding new RAG functionality, create focused, single-purpose functions that can be composed together.
16. **Authentication** - Use `Depends(get_current_user)` for protected endpoints, `Depends(get_current_user_optional)` for public endpoints with optional auth.
17. **Role-Based Access Control** - Use `dependencies=[Depends(require_roles(["SuperAdmin", "HR"]))]` to restrict endpoints by role.
18. **Bearer Tokens** - Authenticate with `Authorization: Bearer <jwt_token>` header.

**When modifying existing code:**

- Check `utility.py` and `config.py` first for existing utilities and configurations before creating new ones.
- Update `APP_CONTEXT.md` if adding new major features or changing architecture.
- Maintain backward compatibility with existing API endpoints.

---

**Last Updated**: 2025-11-26

## Recent Updates

### 1. Code Quality Refactoring
- Modularized functions in `rag_local_service.py` for better maintainability
- Created dedicated functions: `retrieve_documents()`, `filter_documents_by_rbac()`, `inject_tone_guidance()`, `generate_rag_response()`
- Implemented dependency injection pattern via `app/dependencies.py` for testability

### 2. Authentication & Session Management
- Migrated from API key to JWT token-based authentication
- Session management now uses `session_id` embedded in JWT token (via `user_id`)
- Login endpoint (`/api/auth/token`) returns user profile from `user_meta` table

### 3. Document Parser Integration
- Created `doc_parser` module in `app/utils/` for extensible document parsing
- Supports Markdown, HTML, JSON, and plain text formats
- Integrated into file upload endpoint (`/api/rag/add-file`) for automatic format detection

### 4. Document Versioning System
- Implemented comprehensive version tracking with SQLite database (`version_tracking.py`)
- **Folder-Based Versioning**: Supports `data/{category}/v{version}/*.md` structure (e.g., `data/company/v1/policy.md`)
- Auto-detection of versions from folder paths during seeding
- All document additions now create version 1.0 automatically
- Non-destructive updates create new versions (2.0, 3.0, etc.) while preserving history
- New metadata fields: `document_id`, `version`, `version_created_at`, `version_created_by`, `parent_version`, `status`, `is_latest_version`
- **Reorganized API Endpoints**:
  - `POST /api/rag/documents/add` - Add document (JSON)
  - `POST /api/rag/documents/add-file` - Add document (File)
  - `POST /api/rag/documents/update` - Update document (creates new version)
  - `POST /api/rag/documents/seed` - Seed from data folder
  - `POST /api/rag/documents/clear` - Clear all documents
  - `GET /api/rag/documents/list` - List documents with filtering
  - `GET /api/rag/documents/{document_id}/versions` - Get version history
  - `GET /api/rag/documents/{document_id}/versions/{version}` - Get specific version
  - `GET /api/rag/documents/{document_id}/compare` - Compare two versions with diff
  - `POST /api/rag/documents/{document_id}/archive` - Archive a version
- Version comparison uses Python's `difflib` for unified diffs
- Semantic versioning (1.0, 2.0, 3.0) with auto-increment
- Version status support: draft, pending_approval, published, archived

### 5. Enhanced RBAC (Role-Based Access Control)
- **Metadata-Driven Access Control**: Documents use `.meta.json` companion files for permissions
- **Sensitivity Levels**: `public_internal`, `department_confidential`, `role_confidential`, `highly_confidential`, `personal`
- **Department Validation**: Enforces valid departments (HR, Finance, Engineering, IT, Legal, Executive, Admin, General)
- **Role-Based Creation Restrictions**:
  - Guest/Employee: Can only create `public_internal` documents
  - Manager: Can create `public_internal` + `department_confidential`
  - HR: Can create up to `personal` (except `highly_confidential`)
  - SuperAdmin: Can create ANY sensitivity level
- **Department Ownership**: Users can only update documents from their own department (unless SuperAdmin/HR)
- **Metadata Validation**: Validates `sensitivity`, `department`, `allowed_roles`, and `owner_id` fields
- **Pre-Response Filtering**: `filter_documents_by_rbac()` removes unauthorized content before LLM generation
- **Public Summaries**: Shows fallback summaries when full content is restricted
- **Comprehensive Audit Logging**:
  - `RBAC_ACCESS_DENIED`: Retrieval attempts blocked by permissions
  - `RBAC_UPDATE_DENIED`: Cross-department update attempts
  - `METADATA_VALIDATION_FAILED`: Invalid metadata in creation attempts
  - `DOCUMENT_CREATED`: Successful document creation with metadata
  - `DOCUMENT_UPDATED`: Successful updates with version info
  - `METADATA_CHANGE`: Sensitivity level changes
  - `FILE_UPLOADED`: File upload tracking
- **Seeding Enhancement**: `seed_from_file()` automatically loads metadata from `.meta.json` companion files
- **Smart Versioning**: Automatically filters out older versions from search results if a newer version is accessible to the user (Version Deduplication)
- **Cross-Department Overrides**: `department_confidential` documents can be shared with specific roles (e.g., Managers) from other departments via `allowed_roles`
- **Expanded Role Support**: Added support for `Employee L1` and `Employee L2` roles

