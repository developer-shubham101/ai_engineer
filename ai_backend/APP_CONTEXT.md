# 🚀 Multi-Provider Enterprise RAG System - Technical Context

**Single Source of Truth for AI Assistant Understanding**

> **For AI Assistants**: This file contains complete system architecture, API specifications, and implementation details. Use this as primary context for code generation, debugging, and system understanding.

---

## 1. System Overview

### Purpose
A **production-ready multi-provider RAG system** supporting both **offline-first** (local models) and **cloud-based** (API) LLM providers through a unified architecture. Designed for enterprise environments with comprehensive RBAC, document versioning, and session management.

### Supported Providers
- **Local Models**: Mistral-7B, Phi-2, Llama-3.2, Gemma-2B, Qwen2 (via llama-cpp-python)
- **Cloud APIs**: Google Gemini, OpenAI GPT, Hugging Face Inference API
- **Shared Components**: ChromaDB vectors, MiniLM embeddings, SQLite sessions

### Key Features
- ✅ **Multi-provider LLM support** with unified API
- ✅ **Enterprise RBAC** with flexible role overrides
- ✅ **Document versioning** with non-destructive updates
- ✅ **Session-aware conversations** with profile management
- ✅ **Offline-first architecture** with cloud integration
- ✅ **JWT authentication** with comprehensive audit logging

---

## 2. Architecture Overview

```
User Request → FastAPI Router → Provider Service → Base RAG Service → Response
                                      ↓
                              Shared Components:
                              • ChromaDB (Vector Storage)
                              • MiniLM (Embeddings)
                              • SQLite (Sessions/Users/Versions)
                              • RBAC Filter (Security)
```

### Core Components

**🏗️ Base Layer**
- `base_rag_service.py` - Abstract RAG service with RBAC filtering
- `chroma_utils.py` - Vector database operations
- `utility.py` - Shared utilities and embedding management

**🤖 Provider Layer**
- `rag_local_service.py` - Local model implementation + document management
- `google_models.py` - Google Gemini API integration
- `gpt_rag_service.py` - OpenAI GPT API integration
- `hf_rag_service.py` - Hugging Face API integration

**🔐 Security & Data**
- `auth.py` + `user_service.py` - JWT authentication & user management
- `support_chat.py` - Session management & conversation history
- `version_tracking.py` - Document versioning system
- `profile_analyzer.py` - User profile analysis for personalization

**⚙️ Infrastructure**
- `model_manager.py` - Local LLM loading & caching
- `local_model_manager.py` - Multi-model support and selection
- `prompt_builder.py` - Prompt construction & token management
- `sentiment_classifier.py` - Sentiment analysis

**🌐 API Layer**
- `main.py` - FastAPI application with lifecycle management
- `api_routes_rag.py` - Multi-provider RAG endpoints
- `api_routes_auth.py` - Authentication endpoints
- `api_routes_models.py` - Model management endpoints
- `dependencies.py` - Dependency injection for auth and services

---

## 3. Directory Structure

```
ai_backend/
├── app/
│   ├── services/              # Core business logic
│   │   ├── base_rag_service.py      # Abstract RAG base
│   │   ├── rag_local_service.py     # Local models + docs
│   │   ├── google_models.py         # Google Gemini
│   │   ├── gpt_rag_service.py       # OpenAI GPT
│   │   ├── hf_rag_service.py        # Hugging Face
│   │   ├── auth.py                  # JWT token management
│   │   ├── user_service.py          # User management
│   │   ├── support_chat.py          # Session management
│   │   ├── version_tracking.py      # Document versions
│   │   ├── profile_analyzer.py      # User personalization
│   │   ├── model_manager.py         # LLM loading
│   │   ├── local_model_manager.py   # Multi-model support
│   │   ├── prompt_builder.py        # Prompt construction
│   │   ├── sentiment_classifier.py  # Sentiment analysis
│   │   ├── utility.py               # Shared utilities
│   │   └── chroma_utils.py          # Vector DB operations
│   ├── utils/                 # Utility modules
│   │   └── doc_parser/             # Document parsing
│   ├── config/               # Configuration files
│   │   ├── onboarding_fields.json  # User onboarding
│   │   └── local_models.json       # Model configurations
│   ├── main.py               # FastAPI application
│   ├── api_routes_rag.py     # RAG endpoints
│   ├── api_routes_auth.py    # Auth endpoints
│   ├── api_routes_models.py  # Model endpoints
│   ├── dependencies.py       # Dependency injection
│   ├── config.py            # Application configuration
│   └── logging_config.py    # Logging setup
├── database/                # SQLite databases
├── chroma_storage/          # Vector database
├── models/                  # Local LLM files
├── embeddings_models/       # Embedding models
├── data/                   # Seed documents
├── scripts/                # Utility scripts
└── requirements.txt        # Python dependencies
```

---

## 4. API Endpoints

### Authentication (`/api/auth/`)

**POST /api/auth/token** - User login
```json
Request: {"username": "string", "password": "string"}
Response: {
  "access_token": "jwt_token",
  "token_type": "bearer",
  "user": {
    "user_id": "string",
    "username": "string", 
    "role": "string",
    "department": "string",
    "profile": {...}
  }
}
```

### Multi-Provider RAG (`/api/rag/`)

**POST /api/rag/{provider}/query** - Unified query interface
- **Providers**: `local`, `google`, `gpt`, `huggingface`/`hf`
- **Authentication**: Optional (Bearer token for personalization)
- **RBAC**: Automatic filtering based on user role/department

```json
Request: {
  "question": "string",
  "top_k": 3,
  "use_llm": true,
  "max_tokens": 256,
  "category": "string",
  "debug": false,
  "local_llm_model": "phi2"  // Local provider only
}

Response: {
  "answer": "string",
  "retrieved": [{"id": "string", "text": "string", "metadata": {}, "distance": 0.5}],
  "context": "string"
}
```

### Document Management (`/api/rag/documents/`)

**POST /api/rag/documents/add** - Add document (JSON)
**POST /api/rag/documents/add-file** - Upload file
**POST /api/rag/documents/update** - Update document (creates new version)
**POST /api/rag/documents/seed** - Seed from data folder
**GET /api/rag/documents/list** - List documents with filtering
**GET /api/rag/documents/{id}/versions** - Version history
**GET /api/rag/documents/{id}/compare** - Compare versions
**POST /api/rag/documents/{id}/archive** - Archive version

### Model Management (`/api/models/`)

**GET /api/models/list** - List available models
**GET /api/models/best** - Get best available model
**GET /api/models/downloadable** - Models available for download
**POST /api/models/refresh** - Refresh model cache

---

## 5. RBAC System

### Role Hierarchy
```
SuperAdmin (4) → Manager (3) → HR (2) → Employee (1) → Guest (0)
```

### Sensitivity Levels
```
super_confidential (4)    - SuperAdmin only
highly_confidential (3)   - Manager+ level
role_confidential (2)     - HR+ level  
department_confidential (1) - Employee+ in same department
public_internal (0)       - Everyone
personal (1)              - Owner + HR+ level
```

### RBAC Features
- **Level-based validation**: Users can only create documents at their level or below
- **Role overrides**: `allowed_roles` bypasses hierarchy
- **Department restrictions**: Users below HR level can only update their department
- **Personal documents**: Owner access + HR+ level override
- **Pre-response filtering**: Unauthorized content removed before LLM generation

### Example Metadata
```json
{
  "sensitivity": "highly_confidential",
  "department": "HR", 
  "allowed_roles": ["SuperAdmin", "Employee"],
  "owner_id": "user123",
  "public_summary": "HR policy summary"
}
```

---

## 6. Core Service Functions

### BaseRAGService (Abstract)
```python
async def query_rag(question, requester, top_k, use_llm, **kwargs):
    # 1. Retrieve documents from vector DB
    # 2. Apply RBAC filtering
    # 3. Build context with session history
    # 4. Generate response (provider-specific)
    # 5. Return standardized format

def _allowed_by_metadata(metadata, requester):
    # 1. Personal documents: owner or HR+ level
    # 2. Role override (allowed_roles) - bypasses hierarchy  
    # 3. Department matching (for dept_confidential)
    # 4. Level-based access (default hierarchy)
```

### Provider Services
- **LocalRAGService**: Mistral-7B/Phi-2/etc via llama-cpp-python
- **GoogleRAGService**: Gemini API calls
- **GPTRAGService**: OpenAI API calls
- **HuggingFaceRAGService**: HF Inference API

### Authentication Services
```python
# auth.py
def create_access_token(user_data, session_id=None) -> str
def verify_token(token) -> Optional[Dict]

# user_service.py  
def authenticate_user(username, password) -> Optional[Dict]
def get_user_meta(user_id, key) -> Any
def set_user_meta(user_id, key, value) -> None
```

### Document Management
```python
# rag_local_service.py
async def add_document_to_rag_local(text, metadata, requester_id)
async def update_document_version(doc_id, text, metadata, notes, requester_id)

# version_tracking.py
def create_version_record(doc_id, version, chunks, created_by, metadata)
def get_version_history(doc_id) -> List[Dict]
def compare_document_versions(doc_id, v1, v2) -> str
```

### Session Management
```python
# support_chat.py
def create_session(session_id, role, department)
def store_message(session_id, speaker, content)
def fetch_recent_messages(session_id, limit=5) -> List[Dict]
def build_prompt_prefix(requester, history, category) -> str
```

---

## 7. Data Models

### User/Requester
```python
{
  "user_id": str,
  "username": Optional[str], 
  "role": str,  # SuperAdmin, Manager, HR, Employee, Guest
  "department": str  # HR, Finance, Engineering, IT, Legal, Executive
}
```

### Document Metadata
```python
{
  "source": str,
  "department": str,
  "sensitivity": str,
  "allowed_roles": Optional[List[str]],
  "owner_id": Optional[str],
  "public_summary": Optional[str],
  "document_id": str,
  "version": str,
  "status": str,  # draft, published, archived
  "ingested_at": str,
  "ingested_by": str
}
```

### Session Message
```python
{
  "speaker": str,  # "user" or "assistant"
  "content": str,
  "created_at": str,
  "sentiment": Optional[str],
  "tone": Optional[str]
}
```

---

## 8. Configuration

### Environment Variables
```bash
# Optional cloud API keys
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
HUGGINGFACE_API_TOKEN=your_hf_token

# JWT configuration
JWT_SECRET_KEY=your_secret_key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_DAYS=7

# Model configuration
EMBEDDING_MODEL_KEY=bge-small-en-v1.5
ENABLE_DYNAMIC_MODEL_SELECTION=false
DEFAULT_MODEL_NAME=mistral-7b-instruct-v0.2.Q3_K_M.gguf
```

### Key Settings (config.py)
```python
# Role hierarchy
ROLE_LEVELS = {
    "SuperAdmin": 4, "Manager": 3, "HR": 2, 
    "Employee": 1, "Guest": 0
}

# Sensitivity levels
SENSITIVITY_LEVELS = {
    "super_confidential": 4, "highly_confidential": 3,
    "role_confidential": 2, "department_confidential": 1,
    "public_internal": 0, "personal": 1
}

# Valid departments
VALID_DEPARTMENTS = [
    "HR", "Finance", "Engineering", "IT", 
    "Legal", "Executive", "Admin", "General"
]
```

---

## 9. Local Model Support

### Supported Models
```json
{
  "phi2": "Phi-2 (2.7B) - Default, optimized for reasoning",
  "llama32-1b": "Llama 3.2 1B - Efficient edge model", 
  "llama32-3b": "Llama 3.2 3B - Balanced performance",
  "gemma-2b": "Gemma 2B - Google safety-aligned",
  "qwen2-1.5b": "Qwen2 1.5B - Multilingual 32K context",
  "mistral-7b": "Mistral 7B - Proven performance fallback"
}
```

### Model Management
```python
# local_model_manager.py
def get_available_models() -> List[Dict]
def get_best_available_model() -> Optional[str]
def get_model_info(model_key) -> Dict
```

### Download Scripts
```bash
# List available models
python scripts/download_hf_model.py --list

# Download specific model  
python scripts/download_hf_model.py --download phi2

# Download all models
python scripts/download_hf_model.py --all
```

---

## 10. Logging & Monitoring

### Security Events
- `TOKEN_CREATED` - JWT token generation with context
- `TOKEN_EXPIRED` - Expired token detection
- `INVALID_TOKEN` - Invalid token attempts
- `RBAC_ACCESS_DENIED` - Permission violations
- `RBAC_UPDATE_DENIED` - Cross-department update attempts

### System Events  
- `DOCUMENT_CREATED` - Document creation with metadata
- `DOCUMENT_UPDATED` - Version updates
- `FILE_UPLOADED` - File upload tracking
- `LLM_QUERY_DEBUG` - Query processing details
- `EMBEDDING_MODEL_INIT` - Model loading status

### Debug Logging
- `LLM_FINAL_PROMPT` - Complete prompts (full text)
- `GOOGLE_RESPONSE_TEXT` - Complete responses (full text)
- `LOCAL_RESPONSE_TEXT` - Local model responses
- `EMBEDDING_ENCODE_SUCCESS` - Performance metrics

---

## 11. Development Guidelines

### Code Conventions
- **Python 3.10+** with type hints
- **PEP 8** formatting, 4-space indentation
- **Absolute imports** from `app.` namespace
- **Dependency injection** via FastAPI `Depends`
- **Error handling** with `HTTPException`
- **Logging** with module-level loggers

### Architecture Patterns
- **Template method pattern** in BaseRAGService
- **Singleton pattern** for embedding models and LLM instances
- **Factory pattern** for provider selection
- **Dependency injection** for testability
- **Separation of concerns** (routes/services/utilities)

### Security Best Practices
- **JWT token authentication** with expiration
- **RBAC enforcement** at document level
- **Input validation** and sanitization
- **Audit logging** for all sensitive operations
- **Token preview logging** (truncated for security)

---

## 12. Usage Examples

### Authentication
```bash
# Login
curl -X POST "/api/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```

### Multi-Provider Queries
```bash
# Local model with specific selection
curl -X POST "/api/rag/local/query" \
  -H "Authorization: Bearer <token>" \
  -d '{"question": "What is our policy?", "use_llm": true, "local_llm_model": "phi2"}'

# Google Gemini
curl -X POST "/api/rag/google/query" \
  -H "Authorization: Bearer <token>" \
  -d '{"question": "What is our policy?", "use_llm": true}'

# OpenAI GPT
curl -X POST "/api/rag/gpt/query" \
  -H "Authorization: Bearer <token>" \
  -d '{"question": "What is our policy?", "use_llm": true}'
```

### Document Management
```bash
# Add document
curl -X POST "/api/rag/documents/add" \
  -H "Authorization: Bearer <token>" \
  -d '{"text": "Policy content", "metadata": {"sensitivity": "public_internal", "department": "HR"}}'

# List documents
curl "/api/rag/documents/list?department=HR&status=published"
```

---

## 13. Deployment

### Development
```bash
uvicorn app.main:app --reload --port 5444
```

### Docker
```bash
docker-compose up --build
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 5444
```

### Requirements
- **Python 3.10+**
- **ChromaDB** for vector storage
- **SQLite** for sessions/users/versions
- **Local models** in `models/` directory
- **Embedding models** in `embeddings_models/`

---

## 14. AI Assistant Instructions

**When generating code:**

1. **Use this file as primary context** - All architecture and API details are here
2. **Follow established patterns** - Use existing service structure and dependency injection
3. **Enforce RBAC** - Always check user permissions for document access
4. **Handle errors gracefully** - Use FastAPI `HTTPException` with proper status codes
5. **Use type hints** - Include annotations for all parameters and returns
6. **Import from config** - Use `app/config.py` for constants and settings
7. **Maintain singleton patterns** - Use existing model loading utilities
8. **Log security events** - Use structured logging for audit trails
9. **Validate metadata** - Check sensitivity levels and department restrictions
10. **Support multi-provider** - Ensure code works across all LLM providers

**When debugging:**
- Check RBAC filtering if documents aren't returned
- Verify JWT token format and expiration
- Ensure model files exist in correct directories
- Check database initialization and seeding
- Validate metadata format and sensitivity levels

**When adding features:**
- Update this context file with new functionality
- Maintain backward compatibility with existing APIs
- Add appropriate logging and error handling
- Follow the established service layer pattern
- Include comprehensive type hints and documentation

---

**Last Updated**: 2025-01-10

This context file provides complete system understanding for any AI assistant to effectively work with the codebase, generate accurate code, and maintain system consistency.