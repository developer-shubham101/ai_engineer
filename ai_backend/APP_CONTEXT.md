# 🚀 Multi-Provider Enterprise RAG System - Technical Context

**Single Source of Truth for AI Assistant Understanding**

> **For AI Assistants**: This file contains complete system architecture, API specifications, and implementation details. Use this as primary context for code generation, debugging, and system understanding.

---

## 1. System Overview

### Purpose
A **production-ready multi-provider RAG system** supporting both **offline-first** (local models) and **cloud-based** (API) LLM providers through a unified architecture. Designed for enterprise environments with comprehensive RBAC, document versioning, and session management.

### Supported Providers
- **Local Models**: Auto-selected from local_models.json (Mistral-7B, Phi-2, Llama-3.2, Gemma-2B via llama-cpp-python)
- **Cloud APIs**: Google Gemini-2.5-Flash/Pro, OpenAI GPT-3.5/4, Hugging Face Inference API
- **Shared Components**: ChromaDB vectors, BGE embeddings, SQLite sessions

### Key Features
- ✅ **Multi-provider LLM support** with unified API
- ✅ **Enterprise RBAC** with flexible role overrides
- ✅ **Document versioning** with non-destructive updates
- ✅ **Session-aware conversations** with profile management
- ✅ **Offline-first architecture** with cloud integration
- ✅ **JWT authentication** with comprehensive audit logging
- ✅ **Prompt optimization** with token budgeting and context truncation
- ✅ **Debug capabilities** with final_prompt exposure for optimization
- ✅ **Production-ready** with 8GB+ RAM support for local models
- ✅ **Temperature control** - Unified temperature parameter across all providers

---

## 2. Architecture Overview

```
User Request → FastAPI Router → Container → Modular Services → Response
                                      ↓
                              Modular Components:
                              • Vector DB Module (ChromaDB)
                              • Auth Module (JWT + Users + Sessions)
                              • LLM Module (RAG Orchestrator + Providers)
                              • Core Module (Documents + Versions + Utils)
                              • Config Module (Settings + Constants)
                              • API Module (Models + Handlers)
```

### Modular Architecture

**🏗️ Auth Module** (`app/modules/auth/`)
- `jwt_auth.py` - JWT authentication implementation
- `user_manager.py` - User management with SQLite
- `session_manager.py` - Session management & conversation history
- `rbac.py` - Role-based access control
- `interfaces.py` - Authentication interfaces

**🤖 LLM Module** (`app/modules/llm/`)
- `rag_orchestrator.py` - RAG workflow orchestration
- `providers.py` - LLM provider implementations (Local, Google, GPT, HF)
- `prompt_manager.py` - Optimized prompt construction with token budgeting
- `interfaces.py` - LLM and RAG interfaces

**🗄️ Vector DB Module** (`app/modules/vector_db/`)
- `chroma_impl.py` - ChromaDB implementation
- `embedding_manager.py` - Embedding model management
- `interfaces.py` - Vector database interfaces

**🔧 Core Module** (`app/modules/core/`)
- `document_manager.py` - Document operations
- `version_manager.py` - Document versioning system
- `profile_analyzer.py` - User profile analysis
- `utils.py` - Shared utilities and sentiment analysis

**⚙️ Config Module** (`app/modules/config/`)
- `settings.py` - Environment and application settings
- `constants.py` - System constants and enums
- `models.py` - Configuration data models

**🌐 API Module** (`app/modules/api/`)
- `models.py` - Pydantic request/response models
- `handlers.py` - Request processing logic
- `validators.py` - Input validation

**🌐 API Layer**
- `main.py` - FastAPI application with modular initialization
- `api_routes_rag.py` - RAG endpoints using modular architecture
- `api_routes_auth.py` - Authentication endpoints using container
- `api_routes_models.py` - Model management endpoints
- `dependencies.py` - Dependency injection using container
- `modules/integration.py` - **Dependency injection container**

**🛠️ Utilities**
- `utils/doc_parser.py` - Document parsing utilities

---

## 3. Unified Session Management Architecture

### Design Decision: Custom SQLite vs. LangChain Memory

**Chosen Approach**: Custom SQLite-based session management via `support_chat.py`  
**Alternative Rejected**: LangChain's `ConversationBufferMemory`  
**Decision Date**: Initial architecture (2024), Documented 2025-12-01

#### Why Custom Implementation?

All RAG providers (Local, Google, GPT, HuggingFace) use the **same unified session management system** through inheritance from `BaseRAGService`. This provides:

1. **Persistence**: SQLite storage survives server restarts (critical for production)
2. **Multi-user Support**: Session-isolated storage with unique `session_id` keys
3. **Enterprise Features**: RBAC integration, audit logging, sentiment tracking
4. **Token Optimization**: Precise control over prompt budgets (60-80 token prefixes vs 200+)
5. **Provider Agnostic**: Same session works across local, Google, GPT, HuggingFace providers

#### Session Management Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    BaseRAGService                           │
│  (Unified session management for all providers)             │
│                                                             │
│  inject_personalized_context()                             │
│   ├─ fetch_recent_messages(session_id, limit=2)            │
│   ├─ Extract tone from conversation history                │
│   ├─ get_full_profile(session_id)                          │
│   └─ Build optimized prefix (max 80 tokens)                │
│                                                             │
│  filter_documents_by_rbac()                                 │
│   └─ Version deduplication + RBAC filtering                │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┬──────────────┐
        │                   │                   │              │
        ▼                   ▼                   ▼              ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐  ┌──────────────┐
│ LocalRAG      │  │ GoogleRAG     │  │ GPTRAG        │  │ HuggingFace  │
│ Service       │  │ Service       │  │ Service       │  │ RAGService   │
└───────────────┘  └───────────────┘  └───────────────┘  └──────────────┘
```

#### What All Providers Share

**✅ Session History**
- **Source**: `support_chat.fetch_recent_messages(session_id, limit=2)`
- **Storage**: SQLite (`database/support_sessions.db`)
- **Persistence**: Survives server restarts
- **Isolation**: Per-session (multi-user safe)

**✅ Sentiment & Tone Tracking**
- **Computed**: Automatically on `store_message(session_id, speaker, content)`
- **Used For**: Response adaptation via `build_tone_guidance(tone)`
- **Available To**: All providers through `inject_personalized_context()`

**✅ User Profile Integration**
- **Source**: `get_full_profile(session_id)` or `get_all_user_meta(user_id)`
- **Contains**: Name, position, preferences, department, custom fields
- **Used In**: Personalized prompt prefixes across all providers

**✅ RBAC Filtering**
- **Method**: `filter_documents_by_rbac()` in `BaseRAGService`
- **Rules**: Role hierarchy + department restrictions + `allowed_roles` overrides
- **Applies To**: All retrieved documents before LLM generation

**✅ Token Budgeting**
- **Budget**: 80 tokens maximum for system prefix
- **Strategy**: Prioritize essential info (role > profile > tone > history)
- **Result**: 60-80 token prefixes (vs 200+ with naive approaches)
- **Benefits**: More tokens available for context and generation

**✅ Audit Logging**
- **Events**: `log_user_action()`, `log_security_event()`, `log_performance_metric()`
- **Captured**: User ID, session ID, role, department, query details, provider
- **Storage**: Application logs with structured data

#### Usage Example

```python
# 1. Create session (any provider)
from app.services.legacy.support_chat import create_session, store_message

session_id = create_session(None, role="Employee", department="Engineering")

# 2. Store conversation (manual or automatic)
store_message(session_id, "user", "What is our vacation policy?")
# → Automatically computes sentiment & tone

# 3. Query any provider with same session_id
from app.services.google_models import query_google_rag

result = await query_google_rag(
    query_text="What is our vacation policy?",
    session_id=session_id,  # ← Same session across providers
    requester={"user_id": "u123", "role": "Employee", "department": "Engineering"}
)

# 4. BaseRAGService automatically:
#    - Fetches last 2 messages from SQLite
#    - Extracts user tone
#    - Loads user profile
#    - Builds optimized prefix (60-80 tokens)
#    - Applies RBAC filtering
#    - Logs all actions

# 5. Switch providers seamlessly
from app.services.rag_local_service import query_local_rag

result = await query_local_rag(
    query_text="Can you elaborate?",
    session_id=session_id,  # ← Same session, different provider
    requester={"user_id": "u123", "role": "Employee", "department": "Engineering"}
)
# → Conversation context preserved
```

#### Trade-offs Accepted

**Advantages of Custom System:**
- ✅ Production-ready persistence
- ✅ Multi-user session isolation
- ✅ Enterprise compliance (audit logs)
- ✅ Precise token control
- ✅ Provider-agnostic abstraction

**Disadvantages vs. LangChain:**
- ⚠️ Custom code maintenance
- ⚠️ Less community examples
- ⚠️ No LangChain ecosystem integrations

**Verdict**: Custom implementation provides superior production capabilities at the cost of initial development effort (already completed).

#### Performance Benefits

- **Average prefix size**: 60-80 tokens (vs 200+ with naive approaches)
- **Token savings**: ~60% reduction in system prompt overhead
- **Efficiency ratio**: Measured per-query via `log_performance_metric()`
- **Context availability**: More tokens for retrieved documents and generation

#### Legacy Code Note

`google_models.py` contains legacy LangChain `ConversationBufferMemory` code at the top of the file (marked with clear comments). This code is **NOT used** in the production RAG flow and is kept only for reference. See file comments for detailed explanation.

**References**: 
- Implementation details: `SESSION_UNIFICATION_PLAN.md`
- LangChain comparison: `LANGCHAIN_REVIEW.md`
- Session code: `app/services/support_chat.py`
- Base abstraction: `app/services/base_rag_service.py`

---

## 4. Directory Structure

```
ai_backend/
├── app/
│   ├── modules/              # Modular architecture
│   │   ├── auth/            # Authentication module
│   │   │   ├── jwt_auth.py         # JWT implementation
│   │   │   ├── user_manager.py     # User management
│   │   │   ├── session_manager.py  # Session management
│   │   │   ├── rbac.py            # Role-based access control
│   │   │   └── interfaces.py      # Auth interfaces
│   │   ├── llm/             # LLM module
│   │   │   ├── rag_orchestrator.py # RAG orchestration
│   │   │   ├── providers.py       # LLM providers
│   │   │   ├── prompt_manager.py  # Prompt management
│   │   │   └── interfaces.py      # LLM interfaces
│   │   ├── vector_db/       # Vector database module
│   │   │   ├── chroma_impl.py     # ChromaDB implementation
│   │   │   ├── embedding_manager.py # Embedding management
│   │   │   └── interfaces.py      # Vector DB interfaces
│   │   ├── core/            # Core business logic
│   │   │   ├── document_manager.py # Document operations
│   │   │   ├── version_manager.py  # Document versioning
│   │   │   ├── profile_analyzer.py # User profile analysis
│   │   │   └── utils.py           # Shared utilities
│   │   ├── config/          # Configuration module
│   │   │   ├── settings.py        # Environment settings
│   │   │   ├── constants.py       # System constants
│   │   │   └── models.py          # Config data models
│   │   ├── api/             # API layer components
│   │   │   ├── models.py          # Pydantic models
│   │   │   ├── handlers.py        # Request handlers
│   │   │   └── validators.py      # Input validation
│   │   ├── integration.py   # Dependency injection container
│   │   └── README.md        # Module documentation
│   ├── utils/               # Utility functions
│   │   └── doc_parser.py    # Document parsing
│   ├── api_routes_auth.py   # Authentication endpoints
│   ├── api_routes_rag.py    # RAG endpoints
│   ├── api_routes_models.py # Model management endpoints
│   ├── dependencies.py      # FastAPI dependencies
│   ├── logging_config.py    # Logging configuration
│   └── main.py             # FastAPI application
├── test_module/             # Comprehensive test suite
│   ├── test_authenticator.py    # Authentication tests
│   ├── test_user_manager.py     # User management tests
│   ├── test_session_manager.py  # Session management tests
│   ├── test_vector_store.py     # Vector store tests
│   ├── test_rag_orchestrator.py # RAG orchestrator tests
│   ├── test_runner.py           # Test execution runner
│   ├── conftest.py             # Pytest configuration
│   └── README.md               # Test documentation
├── tests/                   # Legacy test files
├── data/                    # Document storage
│   ├── companyData/         # Company documents
│   ├── examples/            # Example documents
│   └── missions_output/     # Generated content
├── database/                # SQLite databases
├── models/                  # Local LLM models (GGUF)
├── embeddings_models/       # Embedding models
├── chroma_data/            # ChromaDB storage
├── logs/                   # Application logs
├── scripts/                # Utility scripts
├── documents/              # Documentation
├── archive/                # Archived files
└── requirements.txt        # Python dependencies
```

---

## 5. API Endpoints

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
  "temperature": 0.1,  // NEW: Temperature control (0.0-1.0)
  "category": "string",
  "debug": false,
  "local_llm_model": "llama32-1b"  // Local provider only
}

Response: {
  "answer": "string",
  "retrieved": [{"id": "string", "text": "string", "metadata": {}, "distance": 0.5}],
  "context": "string",
  "final_prompt": "string"  // Debug: actual prompt sent to LLM
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

### Model Management (`/api/models/`)

**GET /api/models/list** - List available models
**GET /api/models/best** - Get best available model
**GET /api/models/downloadable** - Models available for download
**POST /api/models/refresh** - Refresh model cache

---

## 6. RBAC System

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

## 7. Temperature Parameter System

### Unified Temperature Control

All RAG providers now support a unified `temperature` parameter that controls response creativity and randomness:

- **Range**: 0.0 (deterministic) to 1.0 (highly creative)
- **Default**: 0.1 (balanced, slightly deterministic)
- **Providers**: Local, Google Gemini, OpenAI GPT, Hugging Face
- **API Integration**: Available in all `/api/rag/{provider}/query` endpoints

### Temperature Behavior by Provider

```python
# Local Models (llama-cpp-python)
- Uses native temperature parameter in llama.cpp
- Applied during _call_llm_with_retry() function
- Supports full 0.0-1.0 range

# Google Gemini API
- Maps to generation_config.temperature
- Applied in GoogleRAGService.generate_response()
- Supports 0.0-1.0 range

# OpenAI GPT API
- Maps directly to OpenAI temperature parameter
- Applied in GPTRAGService.generate_response()
- Supports 0.0-2.0 range (clamped to 1.0 for consistency)

# Hugging Face API
- Maps to parameters.temperature in inference API
- Applied in HuggingFaceRAGService.generate_response()
- Supports 0.0-1.0 range
```

### Implementation Details

**Base RAG Service Integration:**
```python
# BaseRAGService.generate_response() signature
async def generate_response(
    self,
    query_text: str,
    context_text: str,
    final_prefix: str,
    use_llm: bool,
    max_tokens: int,
    temperature: float,  # ← Added to abstract method
    session_id: Optional[str]
) -> Optional[str]
```

**API Request Model:**
```python
class QueryRequest(BaseModel):
    question: str
    top_k: int = 3
    use_llm: bool = False
    max_tokens: int = 256
    temperature: float = 0.1  # Default temperature
    # ... other fields
```

### Usage Examples

```bash
# Deterministic response (temperature = 0.0)
curl -X POST "/api/rag/local/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our policy?", "use_llm": true, "temperature": 0.0}'

# Balanced response (default temperature = 0.1)
curl -X POST "/api/rag/google/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain our benefits", "use_llm": true}'

# Creative response (temperature = 0.7)
curl -X POST "/api/rag/gpt/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Write a summary", "use_llm": true, "temperature": 0.7}'
```

### Testing Framework

A comprehensive test suite validates temperature parameter acceptance:

```python
# tests/test_temperature.py
- Tests temperature values: [0.0, 0.1, 0.5, 1.0]
- Validates all providers accept temperature parameter
- Confirms parameter is passed through to underlying LLM calls
- Handles API key missing scenarios gracefully
```

### Temperature Guidelines

**Recommended Values:**
- **0.0**: Deterministic, factual responses (policies, procedures)
- **0.1**: Default balanced (general Q&A)
- **0.3**: Slightly creative (explanations, summaries)
- **0.5**: Moderate creativity (brainstorming, alternatives)
- **0.7**: High creativity (content generation, ideas)
- **1.0**: Maximum creativity (experimental, highly varied responses)

**Use Cases by Temperature:**
```
0.0-0.2: Compliance, legal, technical documentation
0.2-0.4: Customer support, standard explanations
0.4-0.6: Training content, educational materials
0.6-0.8: Marketing content, creative writing
0.8-1.0: Brainstorming, experimental responses
```

---

## 8. Recent Enhancements (Latest Commits)

### Prompt Optimization System
- **Token Budgeting**: Dynamic allocation between system instructions, context, and user query
- **Context Truncation**: Smart truncation when content exceeds model limits
- **Compressed Prefixes**: Ultra-compact system instructions (60-80 tokens vs 200+ previously)
- **Debug Exposure**: `final_prompt` field in API responses for optimization analysis

### Enhanced Logging & Debugging
- **Performance Metrics**: Token usage, response times, efficiency ratios
- **RBAC Audit Trails**: Detailed logging of access control decisions
- **LLM Interaction Logs**: Complete prompt/response tracking for debugging
- **Sensitive Data Handling**: Secure logging with data classification

### Document Management Improvements
- **Versioned Company Data**: Structured v1/v2 document hierarchy with metadata
- **Enhanced Metadata Schema**: Comprehensive tagging and classification system
- **Archive Management**: Clean separation of current vs archived documents

### Model Management Updates
- **BGE Embeddings**: Switched to 'bge-small-en-v1.5' for better performance
- **Multi-Model Support**: Enhanced local model detection and selection with GGUF format
- **Improved Caching**: Better LLM instance management and reuse
- **Auto-Detection**: Models automatically detected from `models/` directory

### Model Training System (NEW)
- **Llama 3.2 1B Fine-tuning**: Train custom models on company data
- **Automated Export**: Models saved to `models/` directory in GGUF format
- **Background Processing**: Non-blocking training with job tracking
- **Data Filtering**: Automatic exclusion of sensitive documents
- **Format Conversion**: HuggingFace to GGUF conversion for llama.cpp

---

## 8. Model Training Service

### Training Capabilities
- **Base Model**: Llama 3.2 1B (meta-llama/Llama-3.2-1B)
- **Training Data**: Company documents from ChromaDB (filtered by sensitivity)
- **Output Formats**: HuggingFace format + GGUF quantized (Q4_K_M)
- **Training Method**: Instruction tuning with company-specific Q&A pairs

### Training API Endpoints
```python
# Training Management (/api/training/)
GET /api/training/status          # Check training availability
POST /api/training/start          # Start training job (SuperAdmin)
GET /api/training/jobs/{id}       # Monitor training progress
GET /api/training/models          # List trained models
DELETE /api/training/models/{name} # Delete trained model
```

### Training Process
1. **Data Preparation**: Extract non-sensitive documents from ChromaDB
2. **Format Conversion**: Create instruction-tuning format (user/assistant pairs)
3. **Model Training**: Fine-tune Llama 3.2 1B with company data
4. **Export**: Save in HuggingFace format to `models/{name}/`
5. **GGUF Conversion**: Convert to quantized GGUF for llama.cpp integration
6. **Integration**: Trained model automatically available in RAG system

### Training Configuration
```python
{
  "output_name": "llama-3.2-1b-company-tuned",
  "max_samples": 1000,           # Training samples from documents
  "epochs": 3,                   # Training epochs
  "learning_rate": 2e-5,         # Learning rate
  "batch_size": 1,               # Per-device batch size
  "quantization": "q4_k_m"       # GGUF quantization level
}
```

---

## 9. Core Service Functions

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
- **LocalRAGService**: Mistral-7B/Phi-2/etc via llama-cpp-python (with temperature)
- **GoogleRAGService**: Gemini API calls (with temperature)
- **GPTRAGService**: OpenAI API calls (with temperature)
- **HuggingFaceRAGService**: HF Inference API (with temperature)

### Authentication Services (Modular)
```python
# modules/auth/jwt_auth.py
class JWTAuthenticator(IAuthenticator):
    async def authenticate(username, password) -> Optional[Dict]
    async def create_token(user_data, session_id=None) -> str
    async def verify_token(token) -> Optional[Dict]

# modules/auth/user_manager.py
class SQLiteUserManager(IUserManager):
    async def get_user(user_id) -> Optional[Dict]
    async def create_user(user_data) -> str
    async def get_user_metadata(user_id, key) -> Any
    async def set_user_metadata(user_id, key, value) -> bool

# modules/integration.py
container = get_container()
authenticator = container.get_authenticator()
user_manager = container.get_user_manager()
```

### Document Management (Modular)
```python
# modules/core/document_manager.py
class DocumentManager:
    async def add_document(text, metadata, user) -> str
    async def update_document(doc_id, text, metadata, user) -> str
    async def get_document(doc_id) -> Optional[Dict]

# modules/core/version_manager.py
class VersionManager:
    async def create_version(doc_id, content, metadata) -> str
    async def get_versions(doc_id) -> List[Dict]
    async def get_version(doc_id, version_id) -> Optional[Dict]

# modules/integration.py
container = get_container()
doc_manager = container.get_document_manager()
version_manager = container.get_version_manager()
```

### Session Management (Modular)
```python
# modules/auth/session_manager.py
class SQLiteSessionManager(ISessionManager):
    async def create_session(user_id, role, department) -> str
    async def store_message(session_id, speaker, content) -> int
    async def get_messages(session_id, limit=10) -> List[Dict]
    async def set_profile_value(session_id, key, value) -> None
    async def get_full_profile(session_id) -> Dict[str, str]

# modules/integration.py
container = get_container()
session_manager = container.get_session_manager()
```

---

## 10. Data Models

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

## 11. Configuration

### Environment Variables
```bash
# Optional cloud API keys (for cloud providers)
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
HUGGINGFACE_API_TOKEN=your_hf_token

# Server configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Model settings
DEFAULT_MODEL_NAME=mistral-7b-instruct-v0.2
EMBEDDING_MODEL_NAME=bge-small-en-v1.5

# JWT configuration
JWT_SECRET_KEY=your_secret_key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_DAYS=7
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

## 12. Local Model Support

### Supported Models (GGUF Format)
```
models/
├── mistral-7b-instruct-v0.2.Q3_K_M.gguf
├── phi-2-q4_k_m.gguf
├── llama-3.2-1b-instruct-q4_k_m.gguf
├── gemma-2b-it-q4_k_m.gguf
└── ... (auto-detected GGUF files)
```

**Model Capabilities:**
- **Mistral-7B**: Production-ready, balanced performance
- **Phi-2**: Optimized for reasoning tasks
- **Llama-3.2**: Efficient edge models (1B/3B variants)
- **Gemma-2B**: Google safety-aligned model

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

## 13. Logging & Monitoring

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

## 14. Development Guidelines

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

## 15. Usage Examples

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

## 16. Deployment

### Quick Start
```bash
# Clone and install
git clone https://github.com/your-username/ai_backend.git
cd ai_backend
pip install -r requirements.txt

# Start server (works offline with local models)
python -m app.main
```

### Development
```bash
uvicorn app.main:app --reload --port 8000
```

### Production
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Requirements
- **Python 3.10+**
- **ChromaDB** for vector storage
- **SQLite** for sessions/users/versions
- **Local models** in `models/` directory
- **Embedding models** in `embeddings_models/`

---

## 17. Testing & Validation

### Test Infrastructure

**✅ Comprehensive Test Suite:**
- `test_module/` - Professional test suite with modular coverage
- `tests/` - Legacy test files for specific features
- `validate_container_full.py` - Container validation script
- `run_tests.py` - Unified test execution

**✅ Test Module Structure:**
```
test_module/
├── test_authenticator.py    # Authentication module tests
├── test_user_manager.py     # User management tests
├── test_session_manager.py  # Session management tests
├── test_vector_store.py     # Vector store tests
├── test_rag_orchestrator.py # RAG orchestrator tests
├── test_runner.py           # Standalone test runner
├── conftest.py             # Pytest configuration
├── requirements.txt        # Test dependencies
└── README.md               # Test documentation
```

**✅ Test Coverage:**
- **Authentication**: Valid/invalid credentials, edge cases, error handling
- **User Management**: User retrieval, non-existent users, data validation
- **Session Management**: Session creation, message storage, history retrieval
- **Vector Store**: Initialization, accessibility, method validation
- **RAG Orchestrator**: Initialization, type checking, method availability
- **Error Handling**: Comprehensive exception handling across all modules
- **Edge Cases**: None values, empty inputs, invalid parameters

**✅ Dual Execution Support:**
- **Pytest Framework**: Professional testing with fixtures and assertions
- **Standalone Execution**: Direct script execution for quick validation
- **Unified Runner**: Execute all tests or specific modules

### Running Tests

```bash
# Install test dependencies
pip install -r test_module/requirements.txt

# Run all test approaches
python run_tests.py

# Pytest execution (recommended)
pytest test_module/ -v
pytest test_module/test_authenticator.py -v

# Standalone execution
python test_module/test_runner.py
python test_module/test_runner.py auth
python test_module/test_authenticator.py

# Container validation
python validate_container_full.py

# Legacy tests
python tests/test_optimized_prompt.py
python tests/test_rbac_comprehensive.py
```

**✅ Test Features:**
- **Error Handling**: All tests wrapped in try-catch blocks
- **Resource Management**: Proper setup/teardown with pytest fixtures
- **Modular Design**: Each component tested independently
- **Clear Output**: Detailed test results and failure reporting
- **Documentation**: Comprehensive test coverage documentation

## 18. AI Assistant Instructions

**When generating code:**

1. **Use modular architecture** - Import from `app.modules.integration` container
2. **Follow container pattern** - Use `get_container().get_service()` for dependencies
3. **Use interfaces** - Implement abstract base classes from `interfaces.py`
4. **Enforce RBAC** - Always check user permissions through RBAC module
5. **Handle errors gracefully** - Use FastAPI `HTTPException` with proper status codes
6. **Use type hints** - Include annotations for all parameters and returns
7. **Use async/await** - All service methods are async in modular architecture
8. **Log security events** - Use structured logging for audit trails
9. **Validate through services** - Use service layer validation, not direct checks
10. **Container initialization** - Always call `container.initialize()` before use

**When debugging:**
- Check container initialization with `container.initialize()`
- Verify service registration in `integration.py`
- Ensure interfaces are properly implemented
- Check async/await usage in all service calls
- Validate dependency injection flow
- Run test suite to identify issues: `python test_basic_modules.py`

**When adding features:**
- Create interface first in appropriate module
- Implement concrete class following existing patterns
- Register in `integration.py` container
- Update API routes to use container services
- Add comprehensive type hints and async support
- Write tests for new functionality
- Update this context file with new components

---

**Last Updated**: 2025-01-11 (Modular Architecture Complete - Updated Architecture Documentation)

---

## 19. Migration Status Summary

### ✅ **COMPLETED MIGRATION**

**Eliminated Legacy Dependencies:**
- ❌ `services/auth.py` → **DELETED** (replaced by `modules/auth/jwt_auth.py`)
- ❌ `services/user_service.py` → **DELETED** (replaced by `modules/auth/user_manager.py`)
- ❌ `services/support_chat.py` → **DELETED** (replaced by `modules/auth/session_manager.py`)
- ❌ `services/sentiment_classifier.py` → **DELETED** (integrated into `modules/core/utils.py`)
- 📁 `services/utility.py` → **MOVED TO LEGACY**
- 📁 `services/profile_analyzer.py` → **MOVED TO LEGACY**
- 📁 `api_routes_rag.py` (old) → **MOVED TO LEGACY**

**Active Modular Architecture:**
- ✅ `modules/auth/` - Complete authentication system
- ✅ `modules/vector_db/` - Document storage and retrieval
- ✅ `modules/llm/` - RAG orchestration and providers
- ✅ `modules/core/` - Business logic and utilities
- ✅ `modules/config/` - Configuration management
- ✅ `modules/api/` - API layer components
- ✅ `modules/integration.py` - Dependency injection container

**API Endpoints:**
- ✅ `api_routes_auth.py` - Uses modular auth services
- ✅ `api_routes_rag.py` - RAG implementation using modular architecture
- ✅ `api_routes_models.py` - Model management endpoints
- ✅ `dependencies.py` - Container-based dependency injection
- ✅ `main.py` - Modular architecture initialization

**Test Coverage:**
- ✅ **4/4 Basic Module Tests Passed**
- ✅ **Comprehensive Test Suite Available**
- ✅ **API Integration Validated**
- ✅ **Zero Legacy Coupling Confirmed**

### 🎯 **ARCHITECTURE BENEFITS ACHIEVED**

1. **Single Source of Truth** - No duplicate logic
2. **Dependency Injection** - Clean testable services via container
3. **Interface-Based Design** - Proper abstractions
4. **No Legacy Coupling** - All endpoints use modular architecture
5. **Clean Imports** - `from app.modules.integration import get_container`
6. **Comprehensive Test Coverage** - Professional test suite with pytest + standalone execution
7. **Error Handling** - Robust exception handling across all modules
8. **Production Ready** - Validated and tested architecture

**Test Suite Highlights:**
- ✅ **Modular Test Structure** - Separate test files for each component
- ✅ **Dual Execution** - Both pytest and standalone execution supported
- ✅ **Edge Case Coverage** - Tests for None, empty, and invalid inputs
- ✅ **Error Handling** - Comprehensive exception testing
- ✅ **Resource Management** - Proper setup/teardown with fixtures
- ✅ **Clear Documentation** - Detailed test coverage and usage instructions

**Result: Clean, maintainable, comprehensively-tested modular architecture with zero legacy dependencies and professional-grade test coverage.**

This context file provides complete system understanding for any AI assistant to effectively work with the codebase, generate accurate code, and maintain system consistency.