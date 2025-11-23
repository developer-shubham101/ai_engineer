# COPILOT_CONTEXT.md

**Single Source of Truth for Project Context**

> **Instructions for Copilot**: When generating code, prioritize the information in this file. Use only this file + the currently open buffer for context. Do not read the entire repository unless explicitly asked.

---

## 1. Project Summary

A **fully local, CPU-only enterprise RAG (Retrieval Augmented Generation) system** designed for learning and offline testing. The system simulates an enterprise AI assistant for a fictional company with role-based access control (RBAC), document retrieval, sentiment analysis, and multi-turn chat sessions. Everything runs locally without external API dependencies using Mistral-7B-Instruct (GGUF), MiniLM embeddings, ChromaDB, and FastAPI.

---

## 2. High-Level Architecture

```
User Request → FastAPI → RAG Pipeline → RBAC Filter → Local LLM → Response
                                    ↓
                            Chroma Vector DB
                                    ↓
                        Local Embeddings (MiniLM)
```

### Main Components

- **Backend**: FastAPI application (`app/main.py`, `app/api_routes_local.py`)
- **RAG Services**: 
  - `rag_local_service.py` - Core RAG with RBAC, local embeddings, LLM integration
  - `rag_manual_service.py` - Simplified wrapper for CLI/scripts
  - `rag_service.py` - Legacy Google-based RAG (optional)
- **Vector DB**: ChromaDB (persistent storage in `app/chroma_storage/`)
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2) - loaded from `embeddings_models/`
- **Local LLM**: Mistral-7B-Instruct-v0.2.Q3_K_M.gguf via llama-cpp-python (auto-detected from `models/`)
- **Support Chat**: SQLite-based session management (`app/data/support_sessions.db`)
- **Sentiment Analysis**: Local classifier using scikit-learn + sentence-transformers
- **Auth**: Simple API key mapping (`app/services/auth.py`)
- **Utilities**: Centralized paths, constants, shared functions (`app/services/utility.py`)

### Key Directories

- `app/` - Main application code
  - `services/` - Business logic modules
  - `config/` - Configuration files (e.g., `onboarding_fields.json`)
  - `data/` - SQLite databases, seed data
  - `chroma_storage/` - ChromaDB persistence
- `data/` - Document examples, company overview, seed files
- `models/` - Local LLM model files (*.gguf, *.ggml, *.bin)
- `embeddings_models/` - Local embedding model cache
- `sentiment/` - Sentiment classifier artifacts
- `scripts/` - Utility scripts (seed, test, download)
- `docs/` - Documentation

---

## 3. Key APIs / Functions

### Main API Endpoints (`/api/local/*`)

- `POST /api/local/query` - RAG query with RBAC filtering, optional LLM generation, session-aware
- `POST /api/local/add` - Add document via JSON (text + metadata)
- `POST /api/local/add-file` - Upload text file (≤5MB) for ingestion
- `POST /api/local/seed` - Seed default `data/company_overview.txt`
- `POST /api/local/clear` - Clear Chroma collection (Executive/Legal only)
- `POST /api/local/session/start` - Create support chat session
- `POST /api/local/session/end` - End support chat session
- `POST /api/local/sentiment` - Analyze sentiment/tone of text
- `GET /api/local/sentiment/stats` - Get sentiment statistics

### Core Service Functions

**RAG Services** (`rag_local_service.py`):
- `initialize_local_rag()` - Initialize Chroma client and collection
- `add_document_to_rag_local(source_name, text, metadata)` - Chunk, embed, store document
- `query_local_rag(query_text, n_results, requester, llm_prompt_prefix, use_llm, max_tokens, session_id)` - Query with RBAC
- `seed_from_file(file_path, source_name)` - Seed from file or directory
- `clear_collection()` - Delete all documents

**Support Chat** (`support_chat.py`):
- `create_session(session_id, role, department)` - Create new session
- `store_message(session_id, speaker, content)` - Store message with sentiment analysis
- `fetch_recent_messages(session_id, limit)` - Get conversation history
- `get_full_profile(session_id)` - Get user profile data
- `get_next_missing_profile_key(session_id)` - Get next onboarding question
- `set_profile_value(session_id, key, value)` - Set profile field
- `build_prompt_prefix(requester, history, category)` - Build LLM prefix with context

**Utilities** (`utility.py`):
- `get_embedding_model_instance()` - Singleton embedding model loader
- `embed_texts(texts)` - Embed list of texts
- `chunk_text_basic(text, chunk_size, overlap)` - Text chunking
- `sanitize_metadata_dict(meta)` - Sanitize metadata for Chroma
- `build_tone_guidance(tone)` - Generate tone-aware LLM instructions
- `get_local_embedding_model_path()` - Get embedding model path
- `get_data_path(filename)`, `get_config_path(filename)` - Path helpers

**Chroma Utils** (`chroma_utils.py`):
- `ensure_chroma_client(persist_directory, collection_name)` - Get/create Chroma client
- `add_documents_to_collection(collection, documents, metadatas, ids, embeddings)` - Add docs
- `query_collection(collection, query_embeddings, query_texts, n_results)` - Query vector DB
- `delete_all_documents(collection, client, collection_name)` - Clear collection

**Auth** (`auth.py`):
- `get_user_from_api_key(key)` - Map API key to user dict `{user_id, role, department}`

**Sentiment** (`sentiment_classifier.py`):
- `get_global_sentiment()` - Get singleton classifier
- `SentimentToneClassifier.predict_single(text)` - Predict sentiment and tone

---

## 4. Coding Conventions

- **Language**: Python 3.10+ (Docker uses 3.11, local dev may be 3.10)
- **Type Hints**: Use type hints for function parameters and return types (`from __future__ import annotations`)
- **Formatting**: Follow PEP 8; use 4 spaces for indentation
- **Imports**: 
  - Use absolute imports from `app.` namespace
  - Group: stdlib, third-party, local
  - Import from `utility.py` for shared paths/constants to avoid duplication
- **Function Style**: 
  - Use descriptive names, docstrings for public functions
  - Prefer composition over inheritance
  - Use dependency injection (FastAPI `Depends`) for auth/requester
- **Error Handling**: Use FastAPI `HTTPException` for API errors; log exceptions with context
- **Logging**: Use module-level loggers (`logging.getLogger(__name__)`)
- **Constants**: Define in `utility.py`; import rather than duplicate
- **Circular Imports**: Avoid by importing from `utility.py` for shared constants

---

## 5. Important Files & Their Purpose

- `app/main.py` - FastAPI app entry point, lifespan handlers, endpoint registration
- `app/api_routes_local.py` - Local RAG API routes, request/response models, RBAC enforcement
- `app/services/rag_local_service.py` - Core RAG logic: chunking, embedding, retrieval, RBAC filtering, LLM integration
- `app/services/support_chat.py` - SQLite session management, message storage, profile management, tone guidance
- `app/services/utility.py` - **Centralized utilities**: paths, constants, embedding loader, text processing, metadata sanitization
- `app/services/chroma_utils.py` - ChromaDB wrapper functions for client/collection operations
- `app/services/auth.py` - API key to user mapping (role-based access)
- `app/services/sentiment_classifier.py` - Local sentiment/tone classification using scikit-learn
- `app/services/rag_manual_service.py` - Simplified RAG wrapper for CLI/scripts
- `app/logging_config.py` - Logging configuration
- `app/config/onboarding_fields.json` - Onboarding question definitions
- `requirements.txt` - Python dependencies (FastAPI, ChromaDB, sentence-transformers, llama-cpp-python, etc.)
- `Dockerfile` - Docker image definition (Python 3.11, port 5444)
- `docker-compose.yml` - Docker Compose configuration

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

**Requester (from auth)**:
```python
{
  "user_id": str,
  "role": str,  # Employee, Manager, HR, Legal, Executive, etc.
  "department": str  # Engineering, Finance, HR, Legal, IT, etc.
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
  "text": str,
  "sentiment": str,  # positive, negative, neutral
  "tone": str,  # angry, frustrated, polite, urgent, etc.
  "proba": {
    "sentiment": Dict[str, float],
    "tone": Dict[str, float]
  }
}
```

### RBAC Sensitivity Levels

1. `public_internal` - All authenticated users
2. `department_confidential` - Same department or HR/Legal/Executive
3. `role_confidential` - Specific roles (from `allowed_roles`) or HR/Legal/Executive
4. `highly_confidential` - Legal/Executive only
5. `personal` - Owner or HR/Legal/Executive

---

## 7. Dev Notes

### Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Models** (if not present):
   - Embedding model: Run `scripts/download_embeddings_models.py` or place in `embeddings_models/all-MiniLM-L6-v2/`
   - LLM model: Place `*.gguf` file in `models/` directory (e.g., `mistral-7b-instruct-v0.2.Q3_K_M.gguf`)

3. **Environment Variables** (optional):
   - `.env` file for optional cloud API keys (OpenAI, Google, HuggingFace) - not required for local-only mode

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

- No formal test suite currently
- Use scripts in `scripts/` for manual testing:
  - `scripts/quick_test.py` - Quick API test
  - `scripts/test_sentiment.py` - Sentiment classifier test
  - `scripts/train_sentiment.py` - Train sentiment model
  - `scripts/seed_examples.py` - Seed example documents

### Key Behaviors

- **Lazy Loading**: Embedding model and LLM are loaded on first use (singleton pattern)
- **Auto-Seeding**: On startup, attempts to seed `data/company_overview.txt` if present
- **Session Reset**: `support_chat.init_support_chat_db(reset_on_start=True)` resets DB on app start (change for production)
- **CORS**: Currently allows all origins (`allow_origins=["*"]`) - restrict in production
- **File Upload Limit**: 5MB max for `/api/local/add-file`

### Directory Structure Requirements

- `app/chroma_storage/` - Created automatically by ChromaDB
- `app/data/` - Created automatically for SQLite DBs
- `models/` - Must exist with at least one `*.gguf`, `*.ggml`, or `*.bin` file for LLM
- `embeddings_models/all-MiniLM-L6-v2/` - Embedding model (auto-downloaded if missing)
- `sentiment/` - Created automatically for sentiment artifacts

---

## 8. Instructions For Copilot

**When generating code:**

1. **Prioritize this file** - Use information from `COPILOT_CONTEXT.md` as the primary source of truth
2. **Use only this file + currently open buffer** - Do not read entire repository unless explicitly requested
3. **Import from `utility.py`** - Always import shared paths, constants, and utilities from `app/services/utility.py` to avoid duplication
4. **Follow architecture patterns** - Maintain separation: routes in `api_routes_local.py`, business logic in `services/`, utilities centralized
5. **Respect RBAC** - Always enforce sensitivity levels when accessing documents
6. **Use type hints** - Include type annotations for all function parameters and returns
7. **Handle errors gracefully** - Use FastAPI `HTTPException` for API errors; log with context
8. **Maintain singleton patterns** - Use `get_embedding_model_instance()` from `utility.py` for embeddings
9. **Avoid circular imports** - Import shared constants from `utility.py`, not from other service modules
10. **Keep it local-first** - Prefer local solutions; cloud APIs are optional

**When modifying existing code:**

- Check `utility.py` first for existing utilities before creating new ones
- Update `COPILOT_CONTEXT.md` if adding new major features or changing architecture
- Maintain backward compatibility with existing API endpoints
- Follow existing patterns for error handling and logging

---

**Last Updated**: 2025-01-XX (Update this when making significant changes)

