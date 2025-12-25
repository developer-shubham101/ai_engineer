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
- **ColabLLM**: Custom models via /ask endpoint
- **Shared Components**: Configurable Vector Store (ChromaDB or FAISS), BGE embeddings, SQLite sessions

### Key Features
- ✅ **Multi-provider LLM support** with unified API
- ✅ **Enterprise RBAC** with flexible role overrides
- ✅ **Document versioning** with non-destructive updates
- ✅ **Session-aware conversations** with profile management
- ✅ **Persistent conversation history** with cross-device access
- ✅ **Agentic mode** with step-by-step reasoning capabilities
- ✅ **Multimodal AI capabilities** - Audio, Vision, and Media processing
- ✅ **Agent Framework** - Modular architecture with AutoGen and custom orchestrators
- ✅ **CrewAI Integration** - Multi-agent workflows with debate and research capabilities
- ✅ **Speech-to-Text & Text-to-Speech** with multiple providers
- ✅ **OCR and Image Analysis** with CPU-friendly implementations
- ✅ **Emotion Detection** from audio inputs
- ✅ **LLM-assisted metadata generation** - Semantic enrichment for improved RAG
- ✅ **Offline-first architecture** with cloud integration
- ✅ **JWT authentication** with comprehensive audit logging
- ✅ **Prompt optimization** with token budgeting and context truncation
- ✅ **Debug capabilities** with final_prompt exposure for optimization
- ✅ **Production-ready** with 8GB+ RAM support for local models
- ✅ **Temperature control** - Unified temperature parameter across all providers
- ✅ **Comprehensive RAG logging** - Full pipeline tracking for debugging

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
- `prompt_chain.py` - Chain of Responsibility pattern for dynamic prompt building
- `interfaces.py` - LLM and RAG interfaces

**🗄️ Vector DB Module** (`app/modules/vector_db/`)
- `chroma_impl.py` - ChromaDB implementation
- `faiss_vector_store.py` - FAISS implementation (configurable via `VECTOR_STORE_TYPE` env var)
- `embedding_manager.py` - Embedding model management
- `interfaces.py` - Vector database interfaces

**🔧 Core Module** (`app/modules/core/`)
- `document_manager.py` - Document operations
- `version_manager.py` - Document versioning system
- `profile_analyzer.py` - User profile analysis
- `utils.py` - Shared utilities and sentiment analysis
- `metadata_models.py` - **NEW: Metadata data models for LLM enrichment**
- `metadata_generator.py` - **NEW: LLM-based metadata generation**
- `cleanup_service.py` - **NEW: Document cleanup and enrichment pipeline**


**🤖 CrewAI Module** (`app/modules/crew_ai/`) - **NEW**
- `interfaces.py` - CrewAI interfaces for multi-agent workflows
- `orchestrator.py` - CrewAI orchestrator using official library
- `factory.py` - Factory for creating CrewAI orchestrators
- YAML configuration files in `crew_config/` directory

**🎭 Multimodal Module** (`app/modules/multimodal/`) - **NEW**
- `interfaces.py` - Multimodal processing interfaces
- `file_manager.py` - User file management with RBAC
- `stt_providers.py` - Speech-to-Text providers (Vosk, Whisper)
- `tts_providers.py` - Text-to-Speech providers (pyttsx3, espeak)
- `vision_providers.py` - Vision providers (Tesseract, PaddleOCR)
- `emotion_providers.py` - Emotion detection from audio

**🤖 Agents Module** (`app/modules/agents/`) - **NEW**
- `interfaces.py` - Agent and tool interfaces following SOLID principles
- `orchestrators/` - Agent orchestrator implementations
  - `custom/` - Custom single-agent orchestrator
  - `autogen/` - AutoGen multi-agent orchestrator
- `tools.py` - Tool implementations following SRP
- `factories.py` - Factory pattern for tools and orchestrators
- `utils.py` - Utility classes for mock data and formatting

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
- `api_routes_rag.py` - RAG endpoints with agentic mode support
- `api_routes_auth.py` - Authentication endpoints using container
- `api_routes_conversations.py` - Conversation history with RAG logging
- `api_routes_audio.py` - **NEW: Audio processing (STT, TTS, Emotion)**
- `api_routes_vision.py` - **NEW: Vision processing (OCR, Image Analysis)**
- `api_routes_media.py` - **NEW: Media file serving with RBAC**
- `api_routes_models.py` - Model management endpoints
- `api_routes_agents.py` - **NEW: Agent workflow endpoints**
- `api_routes_crew.py` - **NEW: CrewAI multi-agent workflow endpoints**
- `api_routes_cleanup.py` - **NEW: Document cleanup and metadata enrichment**
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

## 3.5 Conversation History Management (NEW)

### Overview

The system now supports **persistent conversation history** that is decoupled from ephemeral sessions. This enables ChatGPT-like conversation management where users can:
- Access conversation history across different devices
- View and restore previous conversations  
- Continue conversations from where they left off
- Full RAG pipeline logging for debugging and analytics

### Architecture

**Database**: `conversations.db` (separate from `support_sessions.db`)

**Tables**:
1. **conversations** - Conversation metadata
   - `id` (TEXT PRIMARY KEY): Unique conversation ID (conv_xxx)
   - `user_id` (TEXT NOT NULL): Owner of conversation
   - `title` (TEXT): Conversation title (auto-generated or user-set)
   - `created_at` (TEXT): Creation timestamp
   - `updated_at` (TEXT): Last update timestamp
   - `is_archived` (BOOLEAN): Soft delete flag
   - `metadata` (TEXT): Additional JSON metadata

2. **messages** - Messages with comprehensive RAG logging
   - Basic fields: `id`, `conversation_id`, `speaker`, `content`, `created_at`
   - Sentiment: `sentiment`, `tone`, `sentiment_meta`
   - **RAG Pipeline Logging**:
     - `user_query`: Original user question
     - `retrieved_context`: Retrieved documents (JSON)
     - `embeddings_used`: Embedding model metadata (JSON)
     - `llm_prompt`: Final prompt sent to LLM
     - `llm_response_raw`: Raw LLM response
     - `llm_provider`: Provider used (local, google, hf)
     - `llm_model`: Model name
     - `llm_tokens_used`: Token count
     - `llm_temperature`, `llm_max_tokens`: Parameters
     - `retrieved_doc_ids`: Document IDs retrieved (comma-separated)
     - `retrieval_top_k`: Top K parameter
     - `use_documents`, `use_llm`: Flags
     - `processing_time_ms`: Performance metric
     - `error_message`: Error tracking

### Conversation Manager

**Location**: `app/modules/conversation/conversation_manager.py`

**Interface**: `IConversationManager`

**Implementation**: `SQLiteConversationManager`

**Key Methods**:
```python
# Conversation CRUD
async def create_conversation(user_id: str, title: Optional[str]) -> str
async def get_conversation(conversation_id: str, user_id: str) -> Optional[Dict]
async def list_conversations(user_id: str, limit: int, offset: int) -> List[Dict]
async def update_conversation(conversation_id: str, user_id: str, **kwargs) -> bool
async def delete_conversation(conversation_id: str, user_id: str) -> bool

# Message Management
async def add_message(conversation_id: str, speaker: str, content: str, ...) -> int
async def add_rag_message(conversation_id: str, speaker: str, content: str, 
                          user_query: str, retrieved_context: List[Dict], 
                          embeddings_used: Dict, llm_prompt: str, ...) -> int
async def get_messages(conversation_id: str, user_id: str, limit: Optional[int]) -> List[Dict]

# Utilities
async def generate_title(conversation_id: str) -> str
```

### API Endpoints

**Base Path**: `/api/conversations`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/conversations` | List all conversations for user |
| POST | `/api/conversations` | Create new conversation |
| GET | `/api/conversations/{id}` | Get specific conversation |
| PUT | `/api/conversations/{id}` | Update conversation (rename) |
| DELETE | `/api/conversations/{id}` | Delete conversation (soft delete) |
| GET | `/api/conversations/{id}/messages` | Get messages with RAG logging |
| POST | `/api/conversations/{id}/restore` | Restore conversation to session |

### Integration with Authentication

**Login Flow**:
1. User authenticates via `/api/auth/token`
2. System creates new conversation automatically
3. System creates session
4. Returns JWT with `session_id`

**Code**:
```python
# In api_routes_auth.py
conversation_manager = container.get_conversation_manager()
conversation_id = await conversation_manager.create_conversation(
    user_id=user_data["user_id"],
    title="New Conversation"
)
```

### RAG Pipeline Logging

**Purpose**: Every RAG query logs complete pipeline execution for:
- Debugging failed queries
- Performance analysis
- Model comparison
- Conversation replay
- Analytics and metrics

**Logged Data**:
- **Input**: User query, parameters (top_k, temperature, etc.)
- **Retrieval**: Retrieved documents, embeddings used, document IDs
- **Generation**: LLM prompt, raw response, provider, model
- **Performance**: Processing time, token count
- **Errors**: Error messages if query failed

**Usage Example**:
```python
# In RAG query endpoint (future implementation)
await conversation_manager.add_rag_message(
    conversation_id=conversation_id,
    speaker="assistant",
    content=answer,
    user_query=request.question,
    retrieved_context=[{"id": doc.id, "text": doc.text, ...} for doc in docs],
    embeddings_used={"model": "bge-small-en-v1.5", "dimensions": 384},
    llm_prompt=final_prompt,
    llm_response_raw=raw_response,
    llm_provider="local",
    llm_model="phi2",
    llm_temperature=0.1,
    llm_max_tokens=256,
    retrieved_doc_ids=[doc.id for doc in docs],
    retrieval_top_k=3,
    use_documents=True,
    use_llm=True,
    processing_time_ms=1250,
    error_message=None
)
```

### Benefits

✅ **Cross-Device Access**: Conversations tied to user_id, not session_id  
✅ **Persistent History**: Survives session ends and server restarts  
✅ **Complete Audit Trail**: Full RAG pipeline logging for every query  
✅ **Debugging Support**: Replay queries with full context  
✅ **Analytics Ready**: Query performance metrics and model comparison  
✅ **ChatGPT-Like UX**: Familiar conversation management interface  

### Migration from Session-Based Messages

**Old Approach** (session_manager.py):
- Messages stored in `support_sessions.db`
- Tied to ephemeral sessions
- Lost when session ends
- No RAG logging

**New Approach** (conversation_manager.py):
- Messages stored in `conversations.db`
- Tied to user accounts
- Persistent across sessions
- Comprehensive RAG logging

**Backward Compatibility**: Session-based messages still supported for legacy flows. New RAG queries should use conversation manager.


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
│   │   ├── conversation/    # Conversation history module
│   │   │   ├── conversation_manager.py # Conversation management with RAG logging
│   │   │   └── __init__.py        # Module exports
│   │   ├── multimodal/      # Multimodal AI module (NEW)
│   │   │   ├── interfaces.py      # Multimodal interfaces
│   │   │   ├── file_manager.py    # User file management
│   │   │   ├── stt_providers.py   # Speech-to-Text providers
│   │   │   ├── tts_providers.py   # Text-to-Speech providers
│   │   │   ├── vision_providers.py # Vision/OCR providers
│   │   │   ├── emotion_providers.py # Emotion detection
│   │   │   └── __init__.py        # Module exports
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
│   │   │   ├── metadata_models.py  # Metadata data models (NEW)
│   │   │   ├── metadata_generator.py # LLM metadata generation (NEW)
│   │   │   ├── cleanup_service.py  # Document cleanup pipeline (NEW)
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
│   ├── api_routes_conversations.py # Conversation history endpoints
│   ├── api_routes_audio.py  # Audio processing endpoints (NEW)
│   ├── api_routes_vision.py # Vision processing endpoints (NEW)
│   ├── api_routes_media.py  # Media serving endpoints (NEW)
│   ├── api_routes_models.py # Model management endpoints
│   ├── api_routes_cleanup.py # Cleanup and enrichment endpoints (NEW)
│   ├── dependencies.py      # FastAPI dependencies
│   ├── logging_config.py    # Logging configuration
│   └── main.py             # FastAPI application
├── test_module/             # Comprehensive test suite
│   ├── test_authenticator.py    # Authentication tests
│   ├── test_user_manager.py     # User management tests
│   ├── test_session_manager.py  # Session management tests
│   ├── test_vector_store.py     # Vector store tests
│   ├── test_rag_orchestrator.py # RAG orchestrator tests
│   ├── test_metadata_generator.py # Metadata generator tests (NEW)
│   ├── test_cleanup_service.py  # Cleanup service tests (NEW)
│   ├── test_runner.py           # Test execution runner
│   ├── conftest.py             # Pytest configuration
│   └── README.md               # Test documentation
├── tests/                   # Legacy test files
├── data/                    # Document storage
│   ├── company/             # Company documents (source)
│   │   ├── v1/             # Version 1 documents
│   │   └── v2/             # Version 2 documents
│   ├── companyData/         # Legacy company documents
│   ├── examples/            # Example documents
│   └── missions_output/     # Generated content
├── cleaned/                 # Enriched documents (NEW)
│   └── company/            # Cleaned company documents
│       ├── v1/             # Enriched v1 documents
│       └── v2/             # Enriched v2 documents
├── database/                # SQLite databases
├── models/                  # Local LLM models (GGUF)
├── embeddings_models/       # Embedding models
├── user_uploaded_files/     # User multimodal files (NEW)
│   └── {user_id}/          # Per-user file isolation
│       ├── audio_*.wav     # Audio files (STT input, TTS output)
│       ├── image_*.jpg     # Image files (OCR input)
│       └── doc_*.pdf       # Document files
├── vector_db/                 # Main directory for vector databases
│   ├── chroma_db/             # ChromaDB storage
│   └── faiss_db/              # FAISS index storage
├── logs/                   # Application logs
├── scripts/                # Utility scripts
├── documents/              # Documentation
├── archive/                # Archived files
├── requirements.txt        # Python dependencies
├── requirements_multimodal.txt # Multimodal AI dependencies (NEW)
└── validate_container_full.py
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
- **Providers**: `local`, `google`, `gpt`, `huggingface`, `colabllm`/`hf`
- **Authentication**: Optional (Bearer token for personalization)
- **RBAC**: Automatic filtering based on user role/department

```json
Request: {
  "question": "string",
  "conversation_id": "string",  // Required: Links query to conversation
  "top_k": 3,
  "use_documents": true,
  "use_llm": true,
  "use_conversation_history": true,  // Include conversation context
  "enable_agentic_mode": false,  // NEW: Enable step-by-step reasoning
  "max_tokens": 256,
  "temperature": 0.1,  // Temperature control (0.0-1.0)
  "category": "string",
  "debug": false,
  "prompt_template": "string",  // Template selection
  "local_llm_model": "llama32-1b"  // Local provider only
}

Response: {
  "answer": "string",
  "retrieved": [{"id": "string", "text": "string", "metadata": {}, "distance": 0.5}],
  "context": "string",
  "final_prompt": "string"  // Debug: actual prompt sent to LLM
}
```

### Agents (`/api/agents/`) - **NEW**

**POST /api/agents/query** - Execute agent workflows
```json
Request: {
  "question": "What is the status of my tickets?",
  "tools": ["get_user_tickets", "get_ticket_comments"],
  "max_steps": 5,
  "temperature": 0.1,
  "orchestrator_type": "custom",  // "custom" or "autogen"
  "debug": true
}
Response: {
  "answer": "Based on my analysis:\n\nStep 1 (get_user_tickets): Found 2 tickets...",
  "steps": [
    {
      "step": 1,
      "tool": "get_user_tickets",
      "input": "current",
      "result": "Found 2 tickets for user...",
      "timestamp": 1640995200.0
    }
  ],
  "tools_used": ["get_user_tickets"],
  "available_tools": ["search_documents", "get_user_tickets", ...],
  "debug_info": {
    "processing_time_ms": 1250,
    "enabled_tools": ["get_user_tickets"],
    "actual_steps": 1
  }
}
```

**GET /api/agents/status** - Get agent system status
**GET /api/agents/tools** - List available tools
**POST /api/agents/tools/{name}/test** - Test individual tools

### CrewAI Multi-Agent Workflows (`/api/crew/`) - **NEW**

**POST /api/crew/query** - Execute CrewAI multi-agent workflows
```json
Request: {
  "topic": "Should companies adopt remote work policies?",
  "workflow_type": "debate",  // debate, research
  "max_iterations": 3,
  "temperature": 0.7,
  "provider": "local",
  "conversation_id": "conv_123"
}
Response: {
  "result": "# Debate Analysis: Topic\n\n## Advocate Position...\n\n## Critic Position...\n\n## Moderator Analysis...",
  "workflow_type": "debate",
  "agents_used": ["Advocate", "Critic", "Moderator"],
  "iterations": 3,
  "execution_time_ms": 5200,
  "available_workflows": ["debate", "research"]
}
```

**GET /api/crew/status** - Get CrewAI system status
**GET /api/crew/workflows** - List available workflows

**Available Workflows:**
- **debate**: Multi-agent debate with Advocate, Critic, and Moderator
- **research**: Comprehensive research with Researcher, Analyst, and Synthesizer

**Key Features:**
- **Official CrewAI Library**: Uses `crewai.Agent`, `crewai.Task`, `crewai.Crew`
- **YAML Configuration**: Agent and task definitions in `crew_config/`
- **Sequential Processing**: Tasks executed in proper order with context
- **LLM Integration**: Works with local and cloud providers
- **Structured Output**: Formatted results with clear agent contributions

### Audio Processing (`/api/audio/`) - **NEW**

**POST /api/audio/stt** - Speech to Text
```json
Request: FormData with audio file
Parameters: {
  "provider": "vosk|whisper",
  "conversation_id": "string"
}
Response: {
  "success": true,
  "data": {
    "text": "extracted text from audio",
    "provider": "vosk",
    "confidence": 0.8
  },
  "file_path": "user_uploaded_files/user123/audio_conv456_001.wav"
}
```

**POST /api/audio/tts** - Text to Speech
```json
Request: {
  "text": "Text to convert to speech",
  "conversation_id": "string",
  "provider": "pyttsx3|espeak"
}
Response: {
  "success": true,
  "data": {
    "text": "original text",
    "provider": "pyttsx3",
    "duration": 5.2
  },
  "file_path": "user_uploaded_files/user123/tts_conv456_002.wav"
}
```

**POST /api/audio/emotion** - Emotion Detection
```json
Request: FormData with audio file
Response: {
  "success": true,
  "data": {
    "emotion": "positive|neutral|excited|calm",
    "confidence": 0.7,
    "provider": "basic"
  }
}
```

### Vision Processing (`/api/vision/`) - **NEW**

**POST /api/vision/ocr** - Extract Text from Images
```json
Request: FormData with image file
Parameters: {
  "provider": "tesseract|paddleocr",
  "conversation_id": "string"
}
Response: {
  "success": true,
  "data": {
    "text": "extracted text from image",
    "provider": "tesseract",
    "confidence": 0.8
  },
  "file_path": "user_uploaded_files/user123/image_conv456_003.jpg"
}
```

**POST /api/vision/describe** - Image Description
```json
Request: FormData with image file
Response: {
  "success": true,
  "data": {
    "description": "Image: 1920x1080 pixels, RGB mode",
    "width": 1920,
    "height": 1080,
    "mode": "RGB"
  }
}
```

### Media Serving (`/api/media/`) - **NEW**

**GET /api/media/{user_id}/{filename}** - Serve Media Files
- **RBAC**: Users can only access their own files
- **Supported**: Audio (.mp3, .wav), Images (.jpg, .png), Documents (.pdf)
- **Returns**: File with appropriate media type

### Conversation Management (`/api/conversations/`)

**GET /api/conversations** - List user conversations
```json
Response: [{
  "id": "conv_xxx",
  "user_id": "string",
  "title": "string",
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:05:00Z",
  "message_count": 5
}]
```

**POST /api/conversations** - Create new conversation
```json
Request: {"title": "Optional conversation title"}
Response: {
  "id": "conv_xxx",
  "user_id": "string",
  "title": "string",
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z",
  "message_count": 0
}
```

**GET /api/conversations/{id}/messages** - Get conversation messages with RAG logging
```json
Response: [{
  "id": 1,
  "speaker": "user|assistant",
  "content": "string",
  "created_at": "2024-01-01T12:00:00Z",
  "sentiment": "positive",
  "tone": "professional",
  // RAG Pipeline Logging
  "user_query": "original question",
  "retrieved_context": [{"id": "doc1", "text": "..."}],
  "llm_prompt": "final prompt sent to LLM",
  "llm_response_raw": "raw LLM response",
  "llm_provider": "local|google|gpt|hf",
  "llm_model": "model name",
  "llm_temperature": 0.1,
  "processing_time_ms": 1250,
  "error_message": null
}]
```

**PUT /api/conversations/{id}** - Update conversation (rename)
**DELETE /api/conversations/{id}** - Delete conversation (soft delete)
**POST /api/conversations/{id}/restore** - Restore conversation to session

### Prompt Templates (`/api/templates/`) - **NEW**

**POST /api/templates** - Create prompt template with variables
```json
Request: {
  "name": "custom_chat",
  "content": "System: You are a {user_role} assistant.\n\nContext: {source_docs}\n\nQuestion: {user_question}",
  "prompt_variables": "user_role|department|source_docs|user_question"
}
Response: {
  "id": 1,
  "name": "custom_chat",
  "content": "System: You are a {user_role} assistant...",
  "prompt_variables": "user_role|department|source_docs|user_question",
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z"
}
```

**GET /api/templates** - List all templates
**GET /api/templates/{name}** - Get specific template
**PUT /api/templates/{name}** - Update template content and variables
**DELETE /api/templates/{name}** - Delete template

**Key Features:**
- **`prompt_variables` field**: Pipe-separated variable names (e.g., `user_role|department|source_docs`)
- **Explicit variable control**: No regex parsing, better performance
- **Backward compatibility**: Empty `prompt_variables` falls back to automatic detection
- **Supported variables**: `user_role`, `department`, `source_docs`, `user_question`, `history`, `user_profile_summary`, etc.
- **Database migration**: Existing templates automatically get empty `prompt_variables` field

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

### Document Cleanup and Metadata Enrichment (`/api/cleanupdata`) - **NEW**

**POST /api/cleanupdata** - Start cleanup and enrichment pipeline
```json
Request: {
  "force": false  // Force cleanup even if one is in progress
}
Response: {
  "started_at": "2025-12-22T13:47:39Z",
  "completed_at": "2025-12-22T13:48:15Z",
  "status": "completed",
  "total_documents": 18,
  "processed_documents": 18,
  "successful_documents": 18,
  "failed_documents": 0,
  "skipped_documents": 0,
  "total_processing_time_ms": 36250.5,
  "average_processing_time_ms": 2013.9,
  "document_statuses": [
    {
      "source_path": "data/company/v1/CEO_memo.md",
      "document_id": "CEO_memo",
      "status": "success",
      "processing_time_ms": 2150.3,
      "enriched_path": "cleaned/company/v1/CEO_memo.md"
    }
  ],
  "errors": []
}
```

**GET /api/cleanupdata/status** - Check cleanup status
```json
Response: {
  "in_progress": false,
  "report": {
    "status": "completed",
    "total_documents": 18,
    "successful_documents": 18,
    ...
  }
}
```

**GET /api/cleanupdata/preview/{document_id}** - Preview metadata enrichment
```json
Parameters: {
  "document_id": "CEO_memo_strategic_vision",
  "version": "v1"  // Optional, defaults to v1
}
Response: {
  "document_id": "CEO_memo_strategic_vision",
  "version": "v1",
  "source_path": "data/company/v1/CEO_memo_strategic_vision.md",
  "original_metadata": {
    "document_type": "memo",
    "department": "HR",
    "sensitivity": "highly_confidential",
    "tags": ["strategy", "vision", "CEO"]
  },
  "enriched_metadata": {
    "strict": {
      "document_type": "memo",
      "department": "HR",
      "sensitivity": "highly_confidential",
      "source": "CEO_memo_strategic_vision.md",
      "tags": ["strategy", "vision", "CEO"]
    },
    "soft": {
      "summary": "CEO announces Agni 2.0 vision with commitment to carbon neutrality by 2030. Key initiatives include transitioning 50% of delivery fleet to electric vehicles by 2025, migrating data centers to green energy, and reducing plastic waste by 30%.",
      "keywords": ["carbon neutral", "sustainability", "Agni 2.0", "green energy", "electric vehicles", "plastic waste", "environmental commitment"],
      "themes": ["sustainability", "corporate strategy", "environmental responsibility", "green initiatives"],
      "entities": {
        "people": ["Aisha Sharma"],
        "organizations": ["Agni Holdings", "Praxis Global", "Saarthi Infotech", "Agni Pharma"],
        "locations": []
      },
      "generated_at": "2025-12-22T13:47:42Z",
      "llm_model": "phi2",
      "confidence": 0.8
    },
    "enriched_at": "2025-12-22T13:47:42Z",
    "processing_time_ms": 2150.3
  },
  "has_enriched": true
}
```


**GET /api/models/list** - List available models
**GET /api/models/best** - Get best available model
**GET /api/models/downloadable** - Models available for download
**POST /api/models/refresh** - Refresh model cache

---

## 6. LLM-Assisted Metadata Generation System - **NEW**

A comprehensive system for enriching documents with semantic metadata using local LLM before vector database storage.

**Key Components**:
- **Metadata Models**: Strict (system) vs Soft (LLM-generated) separation
- **Metadata Generator**: Token-optimized LLM prompts for consistent extraction
- **Cleanup Service**: Orchestrates scanning, processing, and enrichment pipeline
- **API Endpoints**: `/api/cleanupdata` for pipeline control and preview

**Benefits**:
- Improved retrieval quality with semantic metadata
- Better filtering and explainability
- CPU-efficient (LLM used only at ingestion time)
- Preserves original documents (read-only operation)

**Performance**: 2-5 seconds per document on CPU-only systems

**Documentation**: See [METADATA_GENERATION.md](file:///I:/Workspace/GitHub/ai_engineer/ai_backend/documents/METADATA_GENERATION.md) for complete details.

**Testing**: 17/17 tests passing (8 unit + 9 integration)

---

## 7. RBAC System

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

## 8. Temperature Parameter System

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

## 8. Multimodal AI Features (NEW)

### Overview

The system now supports **multimodal AI capabilities** including audio processing, vision analysis, and media management. These features are designed with loose coupling, dependency injection, and factory patterns for easy extensibility.

### Architecture Principles

**🏗️ Design Patterns**:
- **Factory Pattern**: Easy provider switching (STT, TTS, Vision)
- **Dependency Injection**: Loose coupling between components
- **Interface-Based**: All providers implement common interfaces
- **CPU-Optimized**: Prioritizes CPU-friendly models and libraries

**📁 File Management**:
- **User Isolation**: Files stored in `user_uploaded_files/{user_id}/`
- **Generated Filenames**: `{type}_{conversation_id}_{timestamp}.{ext}`
- **RBAC Security**: Users can only access their own files
- **Automatic Cleanup**: Old files cleaned up after 7 days

### Audio Processing Module

**Speech-to-Text Providers**:
- **Vosk** (Default): CPU-friendly, offline, good accuracy
- **Whisper**: Higher accuracy, slower processing
- **Factory**: `create_stt_provider("vosk|whisper")`

**Text-to-Speech Providers**:
- **pyttsx3** (Default): Cross-platform, offline
- **espeak**: Lightweight, command-line based
- **Factory**: `create_tts_provider("pyttsx3|espeak")`

**Emotion Detection**:
- **Basic Provider**: Uses librosa for audio feature extraction
- **Heuristic Classification**: Simple emotion detection (calm, excited, positive, neutral)
- **Extensible**: Easy to add ML-based emotion models

### Vision Processing Module

**OCR Providers**:
- **Tesseract** (Default): Widely supported, good for printed text
- **PaddleOCR**: Better accuracy, supports multiple languages
- **Factory**: `create_vision_provider("tesseract|paddleocr")`

**Image Analysis**:
- **Basic Description**: Image dimensions, color mode, file info
- **Extensible**: Ready for CLIP, BLIP, or custom vision models

### Workflow Integration

**Typical Multimodal Workflow**:
1. **User uploads audio/image** → Multimodal API processes → Returns extracted data
2. **Frontend receives structured data** → Can display or use in text-based `/query`
3. **Text-based RAG processing** → Uses existing `/api/rag/{provider}/query`
4. **Optional TTS conversion** → AI response converted to audio

**Example Flow**:
```
User Voice → /api/audio/stt → Text → /api/rag/local/query → AI Response → /api/audio/tts → Audio Response
```

### Provider Management

**Easy Model Switching**:
```python
# Configuration-based provider selection
stt_provider = create_stt_provider("whisper")  # Switch to Whisper
tts_provider = create_tts_provider("espeak")   # Switch to espeak
vision_provider = create_vision_provider("paddleocr")  # Switch to PaddleOCR
```

**Model Installation**:
- **Vosk**: Download model to `models/vosk-model-small-en-us-0.15`
- **Whisper**: Auto-downloads on first use
- **Tesseract**: System installation required
- **PaddleOCR**: Auto-downloads models on first use

### Security & RBAC

**File Access Control**:
- Users can only upload/access files in their directory
- Media serving endpoint validates user ownership
- Generated filenames prevent path traversal attacks

**Processing Isolation**:
- Each user's files processed independently
- No cross-user data leakage
- Temporary processing files cleaned up

### Performance Considerations

**CPU Optimization**:
- All providers selected for CPU efficiency
- Lazy loading of models (loaded on first use)
- Configurable processing parameters

**File Management**:
- Automatic cleanup of old files
- Efficient file serving with proper media types
- Minimal memory footprint for file operations

### Extension Points

**Adding New Providers**:
1. Implement provider interface (`ISTTProvider`, `ITTSProvider`, etc.)
2. Add to factory function
3. Update configuration options
4. No changes to API routes required

**Future Enhancements**:
- **CLIP Integration**: Semantic image search
- **BLIP Captioning**: Advanced image descriptions
- **Whisper.cpp**: Faster CPU inference
- **Custom Emotion Models**: ML-based emotion detection

## 9. Agents System (NEW)

### Overview

The system now supports a **dedicated Agents module** with LangChain-style tool orchestration, following SOLID principles and factory patterns. This provides a sandbox environment for agent experimentation with safety constraints.

### Architecture Principles

**🏗️ SOLID Compliance:**
- **Single Responsibility**: Each tool has one clear purpose
- **Open/Closed**: Easy to extend with new tools without modifying existing code
- **Liskov Substitution**: All tools implement ITool interface
- **Interface Segregation**: Separate interfaces for tools, orchestrators, and utilities
- **Dependency Inversion**: Orchestrator depends on abstractions, not concrete tools

**🏭 Factory Pattern:**
- `ToolFactory` - Creates individual tools with proper dependencies
- `AgentOrchestratorFactory` - Creates orchestrators with tool sets
- Easy switching between different tool implementations

**💉 Dependency Injection:**
- Integrated into main container (`integration.py`)
- Tools receive dependencies through constructor injection
- No hard-coded dependencies or singletons

### Available Tools

| Tool | Purpose | Dependencies |
|------|---------|-------------|
| `SearchDocumentsTool` | Search knowledge base | IVectorStore |
| `GetUserTicketsTool` | Get support tickets | MockDataProvider |
| `GetTicketCommentsTool` | Get ticket history | MockDataProvider |
| `AnalyzeDataTool` | Analyze patterns | AnalysisProvider |
| `ResearchDataTool` | Generate metrics | MockDataProvider |
| `SummarizeStatusTool` | Compile information | None |

### Safety Features

**Following MOTIVATION.md Guidelines:**
- ✅ **Hard Step Limit**: Maximum 5 steps (configurable)
- ✅ **Tool Whitelisting**: Only registered tools allowed
- ✅ **No Direct DB Access**: Tools use wrapped services
- ✅ **Sandboxed Execution**: Isolated from production systems
- ✅ **Error Handling**: Graceful failure recovery

### Utility Classes

**Separation of Concerns:**
- `MockDataProvider` - Provides research data (tickets, comments, metrics)
- `AnalysisProvider` - Handles data analysis logic
- `DocumentFormatter` - Formats search results
- `StepFormatter` - Formats execution steps and final answers

### API Integration

**Endpoints:**
- `POST /api/agents/query` - Execute agent workflows
- `GET /api/agents/status` - System status and available tools
- `GET /api/agents/tools` - List all tools with descriptions
- `POST /api/agents/tools/{name}/test` - Test individual tools

**Usage Example:**
```bash
curl -X POST "/api/agents/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the status of my tickets?",
    "tools": ["get_user_tickets", "get_ticket_comments"],
    "max_steps": 3,
    "debug": true
  }'
```

### Research Applications

**Learning Scenarios:**
- Multi-step reasoning patterns
- Tool orchestration strategies
- Error handling and recovery
- Agent decision making

**Extensibility:**
- Easy to add new tools via factory
- Configurable step limits and timeouts
- Pluggable data providers
- Custom workflow patterns

### Benefits

- **Clean Architecture**: SOLID principles ensure maintainability
- **Easy Testing**: Each component can be tested independently
- **Research Friendly**: Safe sandbox for experimentation
- **Production Ready**: Proper error handling and logging
- **Extensible**: Factory pattern makes adding tools trivial

## 10. Agentic Mode Feature (RAG Enhancement)

### Overview

The RAG system supports **agentic mode** - an enhanced reasoning capability that instructs the LLM to provide step-by-step analysis within the existing RAG workflow.

### Implementation

**Location**: `app/modules/llm/rag_orchestrator.py`

**Activation**: Set `enable_agentic_mode: true` in RAG query requests

**Behavior**: Enhances prompts with reasoning instructions:
```python
if request.enable_agentic_mode:
    agentic_prompt = f"{final_prompt}\n\nThink step by step and provide a detailed response with reasoning."
    response = await self.generate_response(agentic_prompt, provider, request.max_tokens, request.temperature)
```

**Difference from Agents System:**
- **Agentic Mode**: Enhanced reasoning within RAG workflow
- **Agents System**: Separate tool orchestration with multi-step execution

### Use Cases

- **RAG Enhancement**: Better reasoning in document-based responses
- **Complex Analysis**: Multi-step problem solving within RAG context
- **Educational Content**: Step-by-step explanations with document context

## 11. Recent Enhancements (Latest Commits)

### Prompt Optimization System
- **Token Budgeting**: Dynamic allocation between system instructions, context, and user query
- **Context Truncation**: Smart truncation when content exceeds model limits
- **Compressed Prefixes**: Ultra-compact system instructions (60-80 tokens vs 200+ previously)
- **Debug Exposure**: `final_prompt` field in API responses for optimization analysis
- **Chain of Responsibility**: Dynamic prompt building with modular handlers for different contexts
- **Dynamic Template Selection**: Context-aware template loading from external files (e.g., `prompt_templates/long_context_template.txt`)
- **Personalization**: Context-aware prompt enhancement based on user profile and session history

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

## 12. Prompt Chain Architecture

### Chain of Responsibility Pattern

The system uses a **Chain of Responsibility pattern** for dynamic prompt building, allowing flexible composition of prompt components based on available context.

### Core Components

**PromptContext** - Data container passed through the chain:
```python
@dataclass
class PromptContext:
    user: Optional[Dict[str, Any]] = None
    question: str = ""
    context: str = ""
    history: str = ""  # Added history field
    category: Optional[str] = None
    session_id: Optional[str] = None
    enhanced_query: str = ""
    system_prompt: str = ""
    user_prompt: str = ""
    final_prompt: str = ""
    metadata: Dict[str, Any] = None
```

**Available Handlers:**
- `SystemPromptHandler` - Builds role-based system instructions
- `PersonalizationHandler` - Adds user-specific personalization
- `SecurityHandler` - Injects security constraints based on role
- `QueryEnhancementHandler` - Enhances queries with user context and sentiment
- `UserPromptHandler` - Formats the final user prompt
- `FinalPromptHandler` - Combines all components into final prompt

### Dynamic Chain Building

The chain is built dynamically based on available context:

```python
def _build_dynamic_chain(self, context: PromptContext) -> PromptHandler:
    handlers = []
    
    # Always start with system
    handlers.append(self.available_handlers['system'])
    
    # Add personalization if user exists
    if context.user:
        handlers.append(self.available_handlers['personalization'])
        handlers.append(self.available_handlers['security'])
    
    # Add query enhancement if we have user or session
    if context.user or context.session_id or context.category:
        handlers.append(self.available_handlers['query_enhancement'])
    
    # Always add user prompt and final
    handlers.append(self.available_handlers['user_prompt'])
    handlers.append(self.available_handlers['final'])
    
    # Chain them together
    for i in range(len(handlers) - 1):
        handlers[i].set_next(handlers[i + 1])
    
    return handlers[0]
```

### Usage Examples

**Basic Query (Guest User):**
```python
chain = PromptChain()
final_prompt = await chain.build_prompt(
    question="What are the company policies?",
    context="Policy documents..."
)
# Chain: System → User Prompt → Final
```

**Authenticated User with Session:**
```python
final_prompt = await chain.build_prompt(
    question="What are my benefits?",
    context="HR documents...",
    user={"role": "Employee", "department": "Engineering"},
    session_id="session_123"
)
# Chain: System → Personalization → Security → Query Enhancement → User Prompt → Final
```

**Enhanced Query for Document Retrieval:**
```python
enhanced_query = await chain.build_enhanced_query(
    question="vacation policy",
    user={"role": "Employee", "department": "HR"},
    session_id="session_123",
    category="benefits"
)
# Result: "vacation policy [User: Employee in HR] [Mood: positive] [Category: benefits]"
```

### Benefits

- **Flexibility**: Add/remove handlers based on context
- **Modularity**: Each handler has single responsibility
- **Extensibility**: Easy to add new prompt enhancement logic
- **Testability**: Each handler can be tested independently
- **Performance**: Only necessary handlers are executed
- **Maintainability**: Clear separation of prompt building concerns

### Session History Integration

The prompt chain automatically includes conversation history when available, enabling context-aware responses:

**How It Works:**
1. `RAGOrchestrator` fetches recent messages from `session_manager.fetch_recent_messages(session_id, limit=5)`
2. History is rendered to string format using `session_manager.render_history(messages)`
3. History is passed to `prompt_chain.build_prompt(history=history_str)`
4. `UserPromptHandler` prepends history to the user prompt if available

**History Format:**
```
Conversation History:
[2024-01-01T12:00:00Z] USER: My name is Alice.
[2024-01-01T12:00:05Z] ASSISTANT: Hello Alice!

Question: What is my name?

Context:
[Retrieved documents...]
```

**Benefits:**
- **Context Retention**: AI remembers previous conversation turns
- **Follow-up Questions**: Users can ask "What about that?" and AI understands references
- **Personalized Responses**: AI can reference earlier user statements
- **Session Continuity**: Works across all LLM providers (Local, Google, GPT, HuggingFace)

### Integration with RAG Services

The prompt chain integrates seamlessly with existing RAG services:

```python
# In BaseRAGService or provider services
from app.modules.llm.prompt_chain import PromptChain

class BaseRAGService:
    def __init__(self):
        self.prompt_chain = PromptChain(session_manager=self.session_manager)
    
    async def build_final_prompt(self, query_text, context, user, session_id):
        return await self.prompt_chain.build_prompt(
            question=query_text,
            context=context,
            user=user,
            session_id=session_id
        )
```

---

## 13. Model Training Service

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

## 14. Core Service Functions

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

## 15. Data Models

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

## 16. Configuration

### Environment Variables
```bash
# Optional cloud API keys (for cloud providers)
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
HUGGINGFACE_API_TOKEN=your_hf_token

# ColabLLM provider (optional)
COLABLLM_BASE_URL=http://localhost:8080
COLABLLM_API_KEY=

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

## 17. Local Model Support

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

## 18. Logging & Monitoring

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

## 19. Development Guidelines

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

## 20. Usage Examples

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

# ColabLLM
curl -X POST "/api/rag/colabllm/query" \
  -H "Authorization: Bearer <token>" \
  -d '{"question": "What is our policy?", "use_llm": true}'
```

### Multimodal Processing
```bash
# Speech to Text
curl -X POST "/api/audio/stt" \
  -H "Authorization: Bearer <token>" \
  -F "file=@voice_question.wav" \
  -F "provider=vosk" \
  -F "conversation_id=conv_123"

# Text to Speech
curl -X POST "/api/audio/tts" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"text": "Here is your answer", "conversation_id": "conv_123", "provider": "pyttsx3"}'

# OCR from Image
curl -X POST "/api/vision/ocr" \
  -H "Authorization: Bearer <token>" \
  -F "file=@document.jpg" \
  -F "provider=tesseract" \
  -F "conversation_id=conv_123"

# Emotion Detection
curl -X POST "/api/audio/emotion" \
  -H "Authorization: Bearer <token>" \
  -F "file=@voice_sample.wav" \
  -F "provider=basic"

# Serve Media File
curl "/api/media/user123/tts_conv_123_1640995200.wav" \
  -H "Authorization: Bearer <token>"
```

### Complete Multimodal Workflow
```bash
# 1. Upload voice question
STT_RESPONSE=$(curl -X POST "/api/audio/stt" \
  -H "Authorization: Bearer <token>" \
  -F "file=@question.wav" \
  -F "conversation_id=conv_123")

# 2. Extract text from response
QUESTION_TEXT=$(echo $STT_RESPONSE | jq -r '.data.text')

# 3. Query RAG system
RAG_RESPONSE=$(curl -X POST "/api/rag/local/query" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"$QUESTION_TEXT\", \"conversation_id\": \"conv_123\", \"use_llm\": true}")

# 4. Convert answer to speech
ANSWER_TEXT=$(echo $RAG_RESPONSE | jq -r '.answer')
curl -X POST "/api/audio/tts" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"$ANSWER_TEXT\", \"conversation_id\": \"conv_123\"}"
```

### Agent Workflows
```bash
# Ticket Status Query
curl -X POST "/api/agents/query" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the status of my tickets?",
    "tools": ["get_user_tickets", "get_ticket_comments"],
    "max_steps": 3,
    "debug": true
  }'

# Document Search with Agent
curl -X POST "/api/agents/query" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Search for vacation policy documents",
    "tools": ["search_documents"],
    "max_steps": 2
  }'

# Research Data Analysis
curl -X POST "/api/agents/query" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Analyze user engagement trends",
    "tools": ["research_data", "analyze_data"],
    "debug": true
  }'

# Test Individual Tool
curl -X POST "/api/agents/tools/search_documents/test" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d "vacation policy"

# Get Agent Status
curl -X GET "/api/agents/status" \
  -H "Authorization: Bearer <token>"
```

### Document Management
```bash
# Add document
curl -X POST "/api/rag/documents/add" \
  -H "Authorization: Bearer <token>" \
  -d '{"text": "Policy content", "metadata": {"sensitivity": "public_internal", "department": "HR"}}'

# List documents
curl "/api/rag/documents/list?department=HR&status=published"

# OCR + Document Addition Workflow
# 1. Extract text from scanned document
OCR_RESPONSE=$(curl -X POST "/api/vision/ocr" \
  -H "Authorization: Bearer <token>" \
  -F "file=@scanned_policy.jpg" \
  -F "conversation_id=conv_123")

# 2. Add extracted text to RAG system
EXTRACTED_TEXT=$(echo $OCR_RESPONSE | jq -r '.data.text')
curl -X POST "/api/rag/documents/add" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d "{\"source_name\": \"Scanned Policy Document\", \"text\": \"$EXTRACTED_TEXT\", \"metadata\": {\"sensitivity\": \"public_internal\", \"department\": \"HR\"}}"
```

---

## 21. Deployment

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

## 22. Testing & Validation

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

## 23. AI Assistant Instructions

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

**Last Updated**: 2025-01-11 (Agents Module Added - SOLID Architecture Implementation)

---

## 24. Migration Status Summary

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
- ✅ `modules/agents/` - **NEW: Agent workflows with SOLID architecture**
- ✅ `modules/integration.py` - Dependency injection container

**API Endpoints:**
- ✅ `api_routes_auth.py` - Uses modular auth services
- ✅ `api_routes_rag.py` - RAG implementation using modular architecture
- ✅ `api_routes_models.py` - Model management endpoints
- ✅ `api_routes_agents.py` - **NEW: Agent workflows with factory pattern**
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