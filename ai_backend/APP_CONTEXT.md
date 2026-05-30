# 🚀 Multi-Provider Enterprise RAG System - Technical Context

**Single Source of Truth for AI Assistant Understanding**

> **For AI Assistants**: This file contains complete system architecture, API specifications, and implementation details. Use this as primary context for code generation, debugging, and system understanding.

---

## 1. System Overview

### Purpose
A **production-ready multi-provider RAG system** supporting both **offline-first** (local models) and **cloud-based** (API) LLM providers through a unified architecture. Designed for enterprise environments with comprehensive RBAC, document versioning, and session management.

### Context & Motivation
This system is built as a **learning playground and reference implementation** for Advanced RAG patterns. It allows developers to:
-   **Experiment** with different LLM providers (Local vs. Cloud) and observe trade-offs.
-   **Understand** complex system design patterns like Dependency Injection in Python/FastAPI.
-   **Study** the implementation of enterprise features like RBAC and precise token management.
-   **Debug** and trace the full RAG pipeline to demystify how retrieval and generation utilize context.

### Supported Providers
- **Local Models**: Auto-selected from local_models.json (Mistral-7B, Phi-2, Llama-3.2, Gemma-2B via llama-cpp-python)
- **Cloud APIs**: Google Gemini-2.5-Flash/Pro, OpenAI GPT-3.5/4, Hugging Face Inference API
- **CustomLLM**: Custom/third-party models via /ask endpoint (preferred for third-party APIs)
- **ColabLLM**: Legacy name for custom APIs (backward compatibility)
- **LlamaServer**: llama-server.exe with OpenAI-compatible API
- **Shared Components**: Configurable Vector Store (ChromaDB or FAISS), BGE embeddings, SQLite sessions

### Provider Endpoints

| Provider | Endpoint | Description | Status |
|----------|----------|-------------|--------|
| Local Models | `local` | GGUF models via llama-cpp-python | ✅ Active |
| OpenAI GPT | `gpt`, `openai` | GPT-3.5, GPT-4 via OpenAI API | ✅ Active |
| Google Gemini | `google` | Gemini-2.5-Flash/Pro via Google AI | ✅ Active |
| Hugging Face | `huggingface`, `hf` | Various models via HF Inference API | ✅ Active |
| CustomLLM | `customllm` | Third-party APIs via /ask endpoint | ✅ Active (Preferred) |
| ColabLLM | `colabllm` | Third-party APIs via /ask endpoint | ✅ Active (Legacy) |
| LlamaServer | `llamaserver` | Local server with OpenAI-compatible API | ✅ Active |

### Key Features
- ✅ **Multi-provider LLM support** with unified API
- ✅ **Complete query preprocessing pipeline** - Normalization, spell correction, synonym expansion, query classification (Tier 1 optimization)
- ✅ **Multi-variant hybrid retrieval** - Searches with all query variants for maximum coverage (Tier 1 optimization)
- ✅ **BM25 hybrid retrieval** for keyword + semantic search (Tier 1 optimization)
- ✅ **Cross-encoder reranking** for improved retrieval quality (Tier 1 optimization)
- ✅ **Paragraph-aware chunking** for semantic coherence (Tier 1 optimization)
- ✅ **Enterprise RBAC** with flexible role overrides
- ✅ **Document versioning** with non-destructive updates
- ✅ **Session-aware conversations** with profile management
- ✅ **Persistent conversation history** with cross-device access
- ✅ **Agentic mode** with step-by-step reasoning capabilities
- ✅ **Agent Tools System** with real-time stock, weather, web search, URL scraping and file operations
- ✅ **Internet Search** - DuckDuckGo (free) or SerpAPI for real-time information in agents
- ✅ **Multimodal AI capabilities** - Audio, Vision, and Media processing
- ✅ **Agent Framework** - Modular architecture with AutoGen and custom orchestrators
- ✅ **CrewAI Integration** - Multi-agent workflows with debate and research capabilities
- ✅ **LlamaServer Integration** - Direct llama-server.exe support with OpenAI-compatible API
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
- ✅ **Agent conversation persistence** - Agent interactions saved to separate `agent_messages` table

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
- `providers/` - LLM provider implementations (Local, Google, GPT, HF, ColabLLM, LlamaServer)
- `services/colabllm_rag_service.py` - ColabLLM RAG service implementation
- `provider_factory.py` - Factory for dynamic provider selection
- `prompt_manager.py` - Optimized prompt construction with token budgeting
- `prompt_chain.py` - Chain of Responsibility pattern for dynamic prompt building
- `prompt_builder.py` - Low-level prompt building utilities
- `template_manager.py` - Database-backed prompt template management
- `langchain_prompt_selector.py` - LangChain-based prompt selection
- `model_manager.py` - Local model loading and caching
- `middleware.py` - LLM request/response middleware
- `colabllm_plugin.py` - ColabLLM provider plugin
- `llamaserver_plugin.py` - LlamaServer provider plugin
- `interfaces.py` - LLM and RAG interfaces
- `prompt_templates/` - External prompt template files (balanced_enterprise, strict_rag, pirate, reasoning_analyst, personalized_chat, ultra_compact)

**🗄️ Vector DB Module** (`app/modules/vector_db/`)
- `chroma_impl.py` - ChromaDB implementation
- `faiss_vector_store.py` - FAISS implementation (configurable via `VECTOR_STORE_TYPE` env var)
- `embedding_manager.py` - Embedding model management
- `reranker.py` - Cross-encoder reranking for improved retrieval quality
- `bm25_index.py` - **NEW: BM25 keyword-based retrieval**
- `hybrid_retrieval.py` - **NEW: Reciprocal Rank Fusion for hybrid search**
- `query_preprocessor.py` - **NEW: Query normalization and spell correction**
- `interfaces.py` - Vector database interfaces

**🔧 Core Module** (`app/modules/core/`)
- `document_manager.py` - Document operations
- `version_manager.py` - Document versioning system
- `profile_analyzer.py` - User profile analysis
- `utils.py` - Shared utilities and sentiment analysis
- `metadata_models.py` - **NEW: Metadata data models for LLM enrichment**
- `metadata_generator.py` - **NEW: LLM-based metadata generation**
- `cleanup_service.py` - **NEW: Document cleanup and enrichment pipeline**


**🤖 CrewAI Module** (`app/modules/agents/orchestrators/crewai/`)
- `interfaces.py` - CrewAI interfaces (`ICrewOrchestrator`, `CrewRequest`, `CrewResponse`)
- `orchestrator.py` - CrewAI orchestrator using official `crewai` library
- `crewai_orchestrator.py` - CrewAI orchestrator implementation
- `travel_workflow.py` - CrewAI travel planning workflow
- `crew_config/agents.yaml` - Agent role/goal/backstory definitions (at project root `crew_config/`)
- `crew_config/tasks.yaml` - Task description/expected_output definitions (at project root `crew_config/`)
- **Workflows**: `debate` (Advocate, Critic, Moderator), `research` (Researcher, Analyst, Synthesizer), `smart_travel_planner`
- **LLM**: llama-server via `CREW_BASE_URL` (OpenAI-compatible, configured in settings)

**🎭 Multimodal Module** (`app/modules/multimodal/`) - **NEW**
- `interfaces.py` - Multimodal processing interfaces
- `file_manager.py` - User file management with RBAC
- `audio_utils.py` - Audio preprocessing utilities
- `stt_providers.py` - Speech-to-Text providers (Vosk, Whisper)
- `tts_providers.py` - Text-to-Speech providers (pyttsx3, espeak)
- `vision_providers.py` - Vision providers (Tesseract, PaddleOCR)
- `emotion_providers.py` - Emotion detection from audio

**🤖 Agents Module** (`app/modules/agents/`)
- `interfaces.py` - Agent and tool interfaces (`IAgentOrchestrator`, `AgentRequest`, `AgentResponse`, `ITool`)
- `factories.py` - `AgentOrchestratorFactory` — creates `autogen`, `custom`, `mcp`, or `crewai` orchestrator
- `agent_runner.py` - Legacy LLM-loop runner (REGISTRY used by `run_agent()` only)
- `tools.py` - ITool implementations (legacy, used by agent_runner only)
- `utils.py` - Utility classes for mock data and formatting
- `function_tools/` - Standalone callable tool implementations
  - `tool_web_search.py`, `tool_web_scraper.py`, `tool_stock.py`, `tool_weather.py`
  - `tool_chart.py`, `tool_file.py`, `tool_travel.py`
- `orchestrators/utils/` - **Shared utilities used by all three orchestrators**
  - `tool_registry.py` - Lazy `get_tool_registry()` mapping 21 tool names → callables
  - `tool_utils.py` - `resolve_tools`, `build_tool_catalog`, `execute_tool`, `execute_tool_calls`, cache helpers
  - `json_utils.py` - `extract_json_object` (fast-path → markdown block → generic → auto-repair)
  - `plan_normalizer.py` - `normalize_tool_plan`, `normalize_travel_tool_plan`, `TRAVEL_TOOL_NAMES`, fallback plans
  - `step_utils.py` - `run_team`, `build_executor_steps`, `merge_steps`
  - `__init__.py` - Re-exports all of the above
- `orchestrators/autogen/` - AutoGen multi-agent orchestrator (uses AutoGen v0.4 agents)
  - `autogen_orchestrator.py` - Thin dispatcher; delegates to workflow modules
  - `workflows/debate.py` - 3-agent debate (Advocate, Critic, Moderator)
  - `workflows/research.py` - 6-agent research pipeline
  - `workflows/smart_assistant.py` - ToolSelector → ToolExecutor → Summarizer
  - `workflows/smart_travel_planner.py` - TravelToolSelector → ToolExecutor → TravelPlanner
  - `workflows/prompt_evaluation.py` - PromptParser → CriteriaJudge → Improver → EvalReporter
- `orchestrators/custom/` - Pure-Python multi-agent orchestrator (no AutoGen dependency)
  - `custom_orchestrator.py` - Thin dispatcher; same 4 workflows as AutoGen
  - `workflows/debate.py` - 3-agent debate via sequential `llm_fn` calls
  - `workflows/research.py` - 6-agent pipeline via sequential `llm_fn` calls
  - `workflows/smart_assistant.py` - ToolSelector(llm_fn) → ToolExecutor → Summarizer(llm_fn)
  - `workflows/smart_travel_planner.py` - TravelToolSelector(llm_fn) → ToolExecutor → TravelPlanner(llm_fn)
- `orchestrators/mcp/` - MCP-backed orchestrator (AutoGen agents + MCP tool transport)
  - `mcp_client.py` - `MCPClient` — `list_tools()`, `call_tool()`, `call_tools_parallel()`
  - `mcp_client_stdio.py` - `MCPClient` stdio transport variant
  - `mcp_orchestrator.py` - `smart_assistant` only; ToolSelector/Summarizer identical to AutoGen
- `orchestrators/crewai/` - CrewAI multi-agent orchestrator
  - `orchestrator.py` - CrewAI orchestrator using official `crewai` library
  - `crewai_orchestrator.py` - CrewAI orchestrator implementation
  - `interfaces.py` - CrewAI-specific interfaces
  - `travel_workflow.py` - CrewAI travel planning workflow

**⚙️ Config Module** (`app/modules/config/`)
- `settings.py` - Environment and application settings
- `constants.py` - System constants and enums
- `models.py` - Configuration data models
- `database_config.py` - Database connection configuration
- `local_models.json` - Local model definitions
- `multimodal_models.json` - Multimodal model configurations
- `onboarding_fields.json` - Guest user onboarding field definitions

**🌐 API Module** (`app/modules/api/`)
- `models.py` - Pydantic request/response models
- `handlers.py` - Request processing logic
- `validators.py` - Input validation

**🌐 API Layer**
- `main.py` - FastAPI application with modular initialization
- `api_routes_rag.py` - RAG endpoints with agentic mode support
- `api_routes_auth.py` - Authentication endpoints using container
- `api_routes_conversations.py` - Conversation history with RAG logging
- `api_routes_audio.py` - Audio processing (STT, TTS, Emotion)
- `api_routes_vision.py` - Vision processing (OCR, Image Analysis)
- `api_routes_media.py` - Media file serving with RBAC
- `api_routes_models.py` - Model management endpoints
- `api_routes_agents.py` - Agent workflow endpoints (autogen, custom, mcp, crewai via unified `/api/agents/query`)
- `api_routes_cleanup.py` - Document cleanup and metadata enrichment
- `api_routes_templates.py` - Prompt template CRUD endpoints
- `dependencies.py` - Dependency injection using container
- `modules/integration.py` - Dependency injection container


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
   - `chat_type` (TEXT NOT NULL): Type of conversation — `rag`, `agent`, or `crew`
   - `title` (TEXT): Conversation title (auto-generated or user-set)
   - `created_at` (TEXT): Creation timestamp
   - `updated_at` (TEXT): Last update timestamp
   - `is_archived` (BOOLEAN): Soft delete flag
   - `metadata` (TEXT): Additional JSON metadata

2. **messages** - RAG conversation messages with full pipeline logging
   - Basic fields: `id`, `conversation_id`, `speaker`, `content`, `created_at`
   - Sentiment: `sentiment`, `tone`, `sentiment_meta`
   - RAG Pipeline: `user_query`, `retrieved_context`, `llm_prompt`, `llm_provider`, `llm_model`, `processing_time_ms`, etc.

3. **agent_messages** - Agent conversation log (completely separate from RAG messages)
   - `conversation_id`, `speaker`, `content`, `created_at`
   - `user_query`: Original question
   - `tools_used`: JSON list of tools that ran
   - `steps`: JSON full step-by-step execution log
   - `orchestrator_type`: `custom` or `autogen`
   - `processing_time_ms`: Execution duration
   - `error_message`: Error if any

### Conversation Manager

**Location**: `app/modules/conversation/conversation_manager.py`

**Interface**: `IConversationManager`

**Implementation**: `SQLiteConversationManager`

**Valid chat_type values** (from `VALID_CHAT_TYPES`): `rag`, `agent`, `crew`

**Key Methods**:
```python
# Conversation CRUD
async def create_conversation(user_id: str, chat_type: str, title: Optional[str]) -> str
async def get_conversation(conversation_id: str, user_id: str) -> Optional[Dict]
async def list_conversations(user_id: str, chat_type: Optional[str], limit: int, offset: int) -> List[Dict]
async def update_conversation(conversation_id: str, user_id: str, **kwargs) -> bool
async def delete_conversation(conversation_id: str, user_id: str) -> bool

# Message Management — RAG
async def add_message(conversation_id: str, speaker: str, content: str, ...) -> int
async def add_rag_message(conversation_id: str, speaker: str, content: str,
                          user_query: str, retrieved_context: List[Dict],
                          embeddings_used: Dict, llm_prompt: str, ...) -> int
async def get_messages(conversation_id: str, user_id: str, limit: Optional[int]) -> List[Dict]

# Message Management — Agent
async def add_agent_message(conversation_id: str, speaker: str, content: str,
                            user_query: str, tools_used: List[str],
                            steps: List[Dict], orchestrator_type: str,
                            processing_time_ms: int, error_message: str) -> int
async def get_agent_messages(conversation_id: str, user_id: str, limit: Optional[int]) -> List[Dict]

# Utilities
async def generate_title(conversation_id: str) -> str
```

### API Endpoints

**Base Path**: `/api/conversations`

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/conversations` | List conversations (optional `?chat_type=` filter) |
| POST | `/api/conversations` | Create new conversation (`chat_type` required) |
| GET | `/api/conversations/{id}` | Get specific conversation |
| PUT | `/api/conversations/{id}` | Update conversation (rename) |
| DELETE | `/api/conversations/{id}` | Delete conversation (soft delete) |
| GET | `/api/conversations/{id}/messages` | Get unified messages (all chat types) |
| POST | `/api/conversations/{id}/restore` | Restore conversation to session |

### Integration with Authentication

**Login Flow**:
1. User authenticates via `/api/auth/token`
2. System creates new conversation automatically (default `chat_type="rag"`)
3. System creates session
4. Returns JWT with `session_id`

**Code**:
```python
# In api_routes_auth.py
conversation_manager = container.get_conversation_manager()
conversation_id = await conversation_manager.create_conversation(
    user_id=user_data["user_id"],
    chat_type="rag",
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
│   │   │   ├── audio_utils.py     # Audio preprocessing utilities
│   │   │   ├── stt_providers.py   # Speech-to-Text providers
│   │   │   ├── tts_providers.py   # Text-to-Speech providers
│   │   │   ├── vision_providers.py # Vision/OCR providers
│   │   │   ├── emotion_providers.py # Emotion detection
│   │   │   └── __init__.py        # Module exports
│   │   ├── llm/             # LLM module
│   │   │   ├── providers/         # Provider implementations
│   │   │   │   ├── local.py       # Local llama-cpp provider
│   │   │   │   ├── openai.py      # OpenAI GPT provider
│   │   │   │   ├── google.py      # Google Gemini provider
│   │   │   │   ├── huggingface.py # Hugging Face provider
│   │   │   │   ├── colabllm.py    # ColabLLM provider
│   │   │   │   ├── llamaserver.py # LlamaServer provider
│   │   │   │   └── providers.py   # Provider registry
│   │   │   ├── services/          # Higher-level LLM services
│   │   │   │   └── colabllm_rag_service.py  # ColabLLM RAG service
│   │   │   ├── prompt_templates/  # External template files
│   │   │   │   ├── balanced_enterprise.txt
│   │   │   │   ├── strict_rag.txt
│   │   │   │   ├── pirate_template.txt
│   │   │   │   ├── reasoning_analyst.txt
│   │   │   │   ├── personalized_chat.txt
│   │   │   │   └── ultra_compact.txt
│   │   │   ├── rag_orchestrator.py # RAG orchestration
│   │   │   ├── provider_factory.py # Provider factory
│   │   │   ├── prompt_manager.py  # Prompt management
│   │   │   ├── prompt_chain.py    # Chain of Responsibility
│   │   │   ├── prompt_builder.py  # Low-level prompt utilities
│   │   │   ├── template_manager.py # DB-backed template management
│   │   │   ├── model_manager.py   # Local model loading
│   │   │   ├── middleware.py      # LLM middleware
│   │   │   ├── colabllm_plugin.py # ColabLLM plugin
│   │   │   ├── llamaserver_plugin.py # LlamaServer plugin
│   │   │   ├── langchain_prompt_selector.py
│   │   │   └── interfaces.py      # LLM interfaces
│   │   ├── agents/          # Agent orchestration module
│   │   │   ├── interfaces.py      # IAgentOrchestrator, AgentRequest, AgentResponse
│   │   │   ├── factories.py       # AgentOrchestratorFactory
│   │   │   ├── agent_runner.py    # Legacy LLM-loop runner (REGISTRY)
│   │   │   ├── tools.py           # ITool implementations (custom orchestrator)
│   │   │   ├── utils.py           # Mock data and formatting helpers
│   │   │   ├── function_tools/    # Standalone callable tools
│   │   │   │   ├── tool_web_search.py
│   │   │   │   ├── tool_web_scraper.py
│   │   │   │   ├── tool_stock.py
│   │   │   │   ├── tool_weather.py
│   │   │   │   ├── tool_chart.py
│   │   │   │   ├── tool_file.py
│   │   │   │   └── tool_travel.py
│   │   │   └── orchestrators/
│   │   │       ├── __init__.py
│   │   │       ├── utils/                   # Shared utilities (all orchestrators)
│   │   │       │   ├── __init__.py
│   │   │       │   ├── tool_registry.py     # Lazy tool-name → callable map (21 tools)
│   │   │       │   ├── tool_utils.py        # resolve_tools, build_tool_catalog, execute_tool_calls, cache
│   │   │       │   ├── json_utils.py        # extract_json_object (4-stage fallback)
│   │   │       │   ├── plan_normalizer.py   # normalize_tool_plan, normalize_travel_tool_plan, TRAVEL_TOOL_NAMES
│   │   │       │   └── step_utils.py        # run_team, build_executor_steps, merge_steps
│   │   │       ├── autogen/                 # AutoGen v0.4 orchestrator
│   │   │       │   ├── __init__.py
│   │   │       │   ├── autogen_orchestrator.py  # Thin dispatcher → WORKFLOW_REGISTRY
│   │   │       │   └── workflows/
│   │   │       │       ├── __init__.py
│   │   │       │       ├── debate.py
│   │   │       │       ├── research.py
│   │   │       │       ├── smart_assistant.py
│   │   │       │       ├── smart_travel_planner.py
│   │   │       │       └── prompt_evaluation.py
│   │   │       ├── custom/                  # Pure-Python orchestrator (no AutoGen)
│   │   │       │   ├── custom_orchestrator.py  # Thin dispatcher → WORKFLOW_REGISTRY
│   │   │       │   └── workflows/
│   │   │       │       ├── __init__.py
│   │   │       │       ├── debate.py
│   │   │       │       ├── research.py
│   │   │       │       ├── smart_assistant.py
│   │   │       │       └── smart_travel_planner.py
│   │   │       ├── mcp/                     # MCP-backed orchestrator
│   │   │       │   ├── __init__.py
│   │   │       │   ├── mcp_client.py        # MCPClient (list_tools, call_tool, call_tools_parallel)
│   │   │       │   ├── mcp_client_stdio.py  # MCPClient stdio transport variant
│   │   │       │   └── mcp_orchestrator.py  # smart_assistant only
│   │   │       └── crewai/                  # CrewAI orchestrator
│   │   │           ├── __init__.py
│   │   │           ├── orchestrator.py      # CrewAI orchestrator (official crewai library)
│   │   │           ├── crewai_orchestrator.py # CrewAI orchestrator implementation
│   │   │           ├── interfaces.py        # CrewAI-specific interfaces
│   │   │           └── travel_workflow.py   # CrewAI travel planning workflow
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
│   │   │   ├── models.py          # Config data models
│   │   │   ├── database_config.py # Database configuration
│   │   │   ├── local_models.json  # Local model definitions
│   │   │   ├── multimodal_models.json # Multimodal configs
│   │   │   └── onboarding_fields.json # Onboarding fields
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
│   ├── api_routes_audio.py  # Audio processing endpoints
│   ├── api_routes_vision.py # Vision processing endpoints
│   ├── api_routes_media.py  # Media serving endpoints
│   ├── api_routes_models.py # Model management endpoints
│   ├── api_routes_agents.py # Agent workflow endpoints (autogen, custom, mcp, crewai)
│   ├── api_routes_cleanup.py # Cleanup and enrichment endpoints
│   ├── api_routes_templates.py # Prompt template endpoints
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
│   ├── globalCompany/       # Financial PDFs (AAPL, AMZN, MSFT, etc.)
│   ├── missions_output/     # Generated content
│   └── cleaned/             # Enriched/cleaned documents
├── data/cleaned/            # Enriched documents (inside data/)
│   └── company/            # Cleaned company documents
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
├── requirements_agents.txt # Agent-specific dependencies
├── requirements_mcp.txt    # MCP-specific dependencies
├── requirements_multimodal.txt # Multimodal AI dependencies
├── requirements_pdf.txt    # PDF parsing dependencies
├── AUTO_GEN.md             # AutoGen orchestrator deep-dive
├── FINAL_IMPLEMENTATION_SUMMARY.md # Implementation summary
├── QUERY_PIPELINE_SUMMARY.md # Query pipeline documentation
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
- **Providers**: `local`, `google`, `gpt`, `openai`, `huggingface`, `hf`, `colabllm`, `customllm`, `llamaserver`
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

### Agents (`/api/agents/`)

**POST /api/agents/query** - Execute agent workflow
```json
Request: {
  "question": "What is Tesla stock price and weather in New York?",
  "workflow": "smart_assistant",   // debate | research | smart_assistant | smart_travel_planner | prompt_evaluation
  "orchestrator_type": "autogen",  // autogen | custom | mcp
  "tools": [],                     // empty = all available
  "max_steps": 5,
  "temperature": 0.1,
  "provider": "local",
  "conversation_id": null,
  "debug": false
}
Response: {
  "answer": "Tesla (TSLA) is currently trading at $247.50...",
  "steps": [
    {"step": 1, "agent": "ToolSelector", "type": "tool_routing", "content": "{...}"},
    {"step": 2, "agent": "ToolExecutor", "type": "tool_execution", "tool": "get_stock_price", "args": {"symbol": "TSLA"}, "duration_ms": 312.4, "cached": false},
    {"step": 3, "agent": "Summarizer",   "type": "reasoning", "content": "Tesla is trading at..."}
  ],
  "tools_used": ["get_stock_price", "get_weather"],
  "available_tools": ["web_search", "scrape_url", ...],
  "available_workflows": ["debate", "research", "smart_assistant", "smart_travel_planner", "prompt_evaluation"],
  "orchestrator_type": "autogen",
  "conversation_id": "conv_xxx",
  "debug_info": {"intent": "STOCK_AND_WEATHER", "confidence": 0.95, "routing_source": "llm", ...}
}
```
- `orchestrator_type`: `autogen` (default) | `custom` | `mcp`
- `workflow` applies to all three orchestrators; `mcp` supports `smart_assistant` only
- `conversation_id` is auto-created if not provided; always returned in response
- Saves both turns (user + assistant) to `agent_messages` table after every call
- `tools` filters which function-based tools are injected (empty = all)
- `debug_info` populated for `smart_assistant` / `smart_travel_planner` workflows

**GET /api/agents/status** - Get agent system status, available orchestrators and tools
```json
Response: {
  "orchestrator_types": {"custom": true, "autogen": true, "mcp": false},
  "tools": [{"name": "web_search", "description": "..."}],
  "status": "active"
}
```

**GET /api/agents/workflows?orchestrator_type=autogen** - List workflows and tools for a given orchestrator
```json
Response: {
  "orchestrator_type": "autogen",
  "workflows": ["debate", "research", "smart_assistant", "smart_travel_planner", "prompt_evaluation"],
  "tools": ["web_search", "scrape_url", ...]
}
```

**GET /api/agents/tools** - List all tools from the shared registry
```json
Response: [
  {"name": "web_search",                 "description": "Search the internet for real-time info..."},
  {"name": "scrape_url",                 "description": "Fetch full content from a URL..."},
  {"name": "get_stock_price",            "description": "Get current stock price..."},
  {"name": "get_stock_history",          "description": "Get historical stock prices..."},
  {"name": "generate_stock_chart",       "description": "Generate stock performance chart..."},
  {"name": "get_crypto_price",           "description": "Get current crypto price..."},
  {"name": "generate_chart",             "description": "Generate generic chart from data..."},
  {"name": "get_weather",                "description": "Get current weather for a city..."},
  {"name": "save_research_report",       "description": "Save structured research report as markdown + JSON sidecar..."},
  {"name": "search_flights",             "description": "Search for flights between two cities..."},
  {"name": "search_hotels",              "description": "Search for hotels at a destination..."},
  {"name": "estimate_trip_budget",       "description": "Estimate total trip budget..."},
  {"name": "search_places",              "description": "Search for tourist attractions..."},
  {"name": "search_restaurants",         "description": "Search for restaurants at a destination..."},
  {"name": "generate_itinerary",         "description": "Generate day-wise travel itinerary..."},
  {"name": "get_local_transport_info",   "description": "Get local transport options..."},
  {"name": "get_distance_between_places","description": "Get distance and travel time between places..."},
  {"name": "generate_trip_summary",      "description": "Generate trip summary with highlights..."},
  {"name": "get_currency_exchange",      "description": "Convert amount between currencies..."},
  {"name": "get_geo_distance",           "description": "Get real straight-line distance via OpenStreetMap..."}
]
```

**POST /api/agents/tools/{tool_name}/test** - Test a specific tool directly
```json
// Single-arg tool
Request body: {"input_data": "AAPL"}
Response: {
  "tool": "get_stock_price",
  "input": "AAPL",
  "result": "AAPL: $150.25",
  "status": "success",
  "source": "function"
}

// Multi-arg tool (save_text_file) — pass JSON string
Request body: {"input_data": "{\"filename\": \"out.txt\", \"content\": \"hello\"}"}
Response: {
  "tool": "save_text_file",
  "input": "{\"filename\": \"out.txt\", \"content\": \"hello\"}",
  "result": "Saved 'out.txt' (5 chars)",
  "status": "success",
  "source": "function"
}
```
- Resolves from shared `get_tool_registry()` — single source of truth
- Multi-arg tools require JSON string as `input_data`; returns HTTP 422 with expected keys if plain string given

**GET /api/agents/conversations/{conversation_id}/messages** - Get agent conversation history
```json
Response: {
  "conversation_id": "conv_xxx",
  "messages": [
    {
      "id": 1, "speaker": "user", "content": "What is the status?",
      "created_at": "2025-01-11T10:00:00Z",
      "tools_used": null, "steps": null, "orchestrator_type": "custom"
    },
    {
      "id": 2, "speaker": "assistant", "content": "Based on my analysis...",
      "created_at": "2025-01-11T10:00:01Z",
      "tools_used": ["get_user_tickets"],
      "steps": [{"step": 1, "tool": "get_user_tickets", ...}],
      "orchestrator_type": "custom", "processing_time_ms": 1250
    }
  ],
  "count": 2
}
```
- Reads from `agent_messages` table — completely separate from RAG `messages` table

### CrewAI Multi-Agent Workflows (`/api/agents/query` with `orchestrator_type="crewai"`)

CrewAI workflows are accessed via the unified `/api/agents/query` endpoint with `orchestrator_type="crewai"`. There is no separate `/api/crew/` route.

**POST /api/agents/query** (CrewAI) - Execute CrewAI multi-agent workflows
```json
Request: {
  "topic": "Should companies adopt remote work policies?",
  "workflow_type": "debate",  // debate | research
  "max_iterations": 3,
  "temperature": 0.7,
  "provider": "local",
  "conversation_id": "conv_123"  // optional, auto-created if omitted
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
- `conversation_id` is auto-created if not provided; saved to DB but **not returned** in response
- Both user and assistant messages saved to `messages` table with `chat_type="crew"`
- Assistant message persists: `workflow_type`, `agents_used`, `iterations`, `processing_time_ms`

**Available CrewAI Workflows:**
- **debate**: Multi-agent debate with Advocate, Critic, and Moderator
- **research**: Comprehensive research with Researcher, Analyst, and Synthesizer
- **smart_travel_planner**: CrewAI travel planning workflow

**Key Features:**
- **Official CrewAI Library**: Uses `crewai.Agent`, `crewai.Task`, `crewai.Crew`
- **YAML Configuration**: Agent and task definitions in `crew_config/agents.yaml` and `crew_config/tasks.yaml`
- **Sequential Processing**: Tasks executed in order with context passed between agents
- **LLM**: llama-server via `CREW_BASE_URL` (OpenAI-compatible endpoint)
- **DB Persistence**: Every query saved to `messages` table (`chat_type="crew"`) via unified agent conversation manager

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
```
Query params: ?chat_type=rag|agent|crew (optional), ?limit=50, ?offset=0
```
```json
Response: [{
  "id": "conv_xxx",
  "user_id": "string",
  "chat_type": "rag",
  "title": "string",
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:05:00Z",
  "message_count": 5
}]
```

**POST /api/conversations** - Create new conversation (`chat_type` is required)
```json
Request: {"chat_type": "rag", "title": "Optional title"}
Response: {
  "id": "conv_xxx",
  "user_id": "string",
  "chat_type": "rag",
  "title": "string",
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:00Z",
  "message_count": 0
}
```

**GET /api/conversations/{id}/messages** - Get unified messages for any chat_type

Messages include a `chat_type` field. Unused fields for a given type are `null`.
```json
Response: {
  "conversation_id": "conv_xxx",
  "chat_type": "rag",
  "messages": [{
    "id": 1,
    "conversation_id": "conv_xxx",
    "chat_type": "rag",
    "speaker": "user|assistant",
    "content": "string",
    "created_at": "2024-01-01T12:00:00Z",
    "processing_time_ms": 1250,
    "error_message": null,
    // RAG fields (populated when chat_type=rag)
    "user_query": "original question",
    "llm_provider": "local|google|gpt|hf",
    "llm_model": "model name",
    "llm_prompt": "final prompt sent to LLM",
    "llm_response_raw": "raw LLM response",
    "llm_tokens_used": 512,
    "llm_temperature": 0.1,
    "llm_max_tokens": 256,
    "retrieval_top_k": 3,
    "use_documents": true,
    "use_llm": true,
    "sentiment": "positive",
    "tone": "professional",
    "retrieved_context": [{"id": "doc1", "text": "..."}],
    "embeddings_used": {"model": "bge-small-en-v1.5"},
    "retrieved_doc_ids": ["doc1"],
    "sentiment_meta": {},
    // Agent fields (populated when chat_type=agent)
    "orchestrator_type": null,
    "tools_used": null,
    "steps": null,
    // Crew fields (populated when chat_type=crew)
    "workflow_type": null,
    "iterations": null,
    "agents_used": null
  }],
  "count": 1
}
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

## 7. Template System - Message-Based Architecture

### Overview

The template system uses **JSON message arrays** instead of string parsing for strict LLM behavior control. Templates are stored in the database and follow a specific structure for optimal message ordering.

### Template Structure

Each template **must** contain exactly 2 messages:
1. **System message** (index 0) - Defines AI behavior and context
2. **User message** (index 1) - Contains the user's question with variables

### Message Assembly Flow

When a template is used, messages are assembled in this order:
1. **System message** (from template[0]) - with variable substitution
2. **Conversation history** (optional) - previous user/assistant exchanges
3. **User message** (from template[1]) - with variable substitution

This ensures all system instructions come first, followed by context (history), and finally the current user query.

### Example Template

```json
{
  "name": "pirate_template",
  "messages": [
    {
      "role": "system", 
      "content": "You are a pirate. Always respond like a pirate with 'Ahoy!' and pirate language. Use words like 'matey', 'arrr', 'ye', 'me hearty' in every response. Never break character - you are always a pirate."
    },
    {
      "role": "user", 
      "content": "{user_question}"
    }
  ],
  "prompt_variables": "user_question"
}
```

### Available Variables

- `{user_question}` - The user's question (required)
- `{source_docs}` - Retrieved document context
- `{user_role}` - User's role (Employee, Manager, etc.)
- `{department}` - User's department
- `{user_profile_summary}` - User profile information
- `{max_tokens}` - Token limit

**Note**: History is automatically inserted between system and user messages, so you don't need a `{history}` variable.

### Message Builder Implementation

The `_build_messages` method in `rag_orchestrator.py` follows this simple flow:

```python
async def _build_messages(template_name, user_question, documents, history, ...):
    # 1. Get template from database
    template_obj = template_manager.get_template(template_name)
    
    # 2. Build system message (template[0]) with variable substitution
    system_msg = template_obj['messages'][0]
    # Replace {user_question}, {source_docs}, etc.
    
    # 3. Add conversation history (optional)
    for msg in history:
        messages.append({"role": msg.speaker, "content": msg.content})
    
    # 4. Build user message (template[1]) with variable substitution
    user_msg = template_obj['messages'][1]
    # Replace {user_question}, {source_docs}, etc.
    
    return messages  # ['system', 'user', 'assistant', 'user', ...]
```

### Provider Compatibility

All providers accept message arrays:
- **LocalLLMProvider**: Converts messages to prompt string
- **OpenAILLMProvider**: Uses messages directly (native support)
- **GoogleLLMProvider**: Converts messages to prompt string
- **HuggingFaceLLMProvider**: Converts messages to prompt string
- **CustomLLMProvider**: Converts messages to prompt string
- **LlamaServerProvider**: Converts to AutoGen core messages

### API Usage

```bash
# Use template in RAG query
curl -X POST "http://localhost:8000/api/rag/local/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the capital of France?",
    "conversation_id": "test123",
    "prompt_template": "pirate_template",
    "use_llm": true
  }'
```

**Expected Response**: *"Ahoy matey! The capital of France be Paris, arrr! That be a fine city for any seafarin' soul to visit, me hearty!"*

### Benefits

1. **No String Parsing**: Direct JSON message arrays
2. **Strict Role Enforcement**: System messages properly separated from user messages
3. **Proper Message Ordering**: System → History → User
4. **Variable Substitution**: Clean variable replacement in both messages
5. **Multi-Provider Support**: Works with all LLM providers
6. **Easy Testing**: Built-in template testing endpoint

### Template Management API

**Base Path**: `/api/templates`

- `POST /api/templates` - Create template
- `GET /api/templates` - List all templates
- `GET /api/templates/{name}` - Get specific template
- `PUT /api/templates/{name}` - Update template
- `DELETE /api/templates/{name}` - Delete template
- `POST /api/templates/test/{name}` - Test template

**Documentation**: See [TEMPLATE_SYSTEM_USAGE.md](file:///I:/Workspace/GitHub/ai_engineer/ai_backend/TEMPLATE_SYSTEM_USAGE.md) for complete details.

---

## 8. RBAC System

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

### Agent Tools System

**Function-Based Tools** (`app/modules/agents/function_tools/`):

| Tool file | Function | Data source | API key required |
|---|---|---|---|
| `tool_stock.py` | `get_stock_price` | ✅ Real — yfinance (Yahoo Finance) | No |
| `tool_stock.py` | `get_stock_history` | ✅ Real — yfinance (Yahoo Finance) | No |
| `tool_stock.py` | `get_crypto_price` | ✅ Real — yfinance (Yahoo Finance) | No |
| `tool_weather.py` | `get_weather` | ✅ Real — OpenWeatherMap API; falls back to demo data if `OPENWEATHER_API_KEY` not set | Optional (`OPENWEATHER_API_KEY`) |
| `tool_web_search.py` | `web_search` | ✅ Real — DuckDuckGo (free, no key) or SerpAPI if `SERPAPI_KEY` set | Optional (`SERPAPI_KEY`) |
| `tool_web_scraper.py` | `scrape_url` | ✅ Real — live HTTP fetch + BeautifulSoup HTML parsing (3000 char limit) | No |
| `tool_chart.py` | `generate_stock_chart` | ✅ Real — yfinance data + matplotlib render; returns demo metadata if matplotlib not installed | No |
| `tool_chart.py` | `generate_chart` | ✅ Real — matplotlib render from structured data; returns demo metadata if matplotlib not installed | No |
| `tool_file.py` | `save_text_file` | ✅ Real — writes to `user_uploaded_files/` | No |
| `tool_file.py` | `save_research_report` | ✅ Real — writes markdown + JSON sidecar to `user_uploaded_files/research_reports/` | No |
| `tool_travel.py` | `get_currency_exchange` | ✅ Real — exchangerate.host API (free, no key) | No |
| `tool_travel.py` | `get_geo_distance` | ✅ Real — OpenStreetMap Nominatim geocoding + Haversine formula | No |
| `tool_travel.py` | `search_flights` | 🔶 Demo — structured simulated data (no live booking API) | No |
| `tool_travel.py` | `search_hotels` | 🔶 Demo — structured simulated data | No |
| `tool_travel.py` | `estimate_trip_budget` | 🔶 Demo — fixed cost estimates per day/traveler | No |
| `tool_travel.py` | `search_places` | 🔶 Demo — hardcoded attractions for known cities (Goa, Jaipur, Kerala, Dubai, Rome, Italy); generic fallback for others | No |
| `tool_travel.py` | `search_restaurants` | 🔶 Demo — structured simulated data | No |
| `tool_travel.py` | `generate_itinerary` | 🔶 Demo — hardcoded day plans for Goa/Jaipur; generic template for others | No |
| `tool_travel.py` | `get_local_transport_info` | 🔶 Demo — generic transport options (auto, taxi, rental bike, bus) | No |
| `tool_travel.py` | `get_distance_between_places` | 🔶 Demo — hardcoded lookup table for common Indian city pairs; falls back to "varies" | No |
| `tool_travel.py` | `generate_trip_summary` | 🔶 Demo — generic highlights and travel tips template | No |

**Status legend**: ✅ Real API/live data — 🔶 Demo/simulated data

**Environment variables for tools**:
```bash
OPENWEATHER_API_KEY=your_key   # Optional: enables real weather data (falls back to demo if not set)
SERPAPI_KEY=your_key           # Optional: upgrades web_search from DuckDuckGo to SerpAPI
```

**Three Orchestrators — Unified Architecture**:

All three orchestrators share the same `WORKFLOW_REGISTRY` pattern and delegate to `orchestrators/utils/` for all tool resolution, execution, JSON parsing, plan normalization, and step building. The only difference is how LLM calls are made:

| Orchestrator | LLM mechanism | Tool execution | Workflows |
|---|---|---|---|
| `autogen` | `AssistantAgent` + `RoundRobinGroupChat` (AutoGen v0.4) | `execute_tool_calls()` via `asyncio.to_thread` | debate, research, smart_assistant, smart_travel_planner, prompt_evaluation |
| `custom` | Plain `async llm_fn(system, user) → str` (LlamaServerProvider) | `execute_tool_calls()` via `asyncio.to_thread` | debate, research, smart_assistant, smart_travel_planner |
| `mcp` | `AssistantAgent` + `RoundRobinGroupChat` (AutoGen v0.4) | `MCPClient.call_tools_parallel()` via MCP stdio | smart_assistant only |
| `crewai` | `crewai.Agent` + `crewai.Crew` (official CrewAI library) | CrewAI task execution | debate, research, smart_travel_planner |

**`WORKFLOW_REGISTRY`** (same for autogen and custom):
| Workflow | Agents |
|---|---|
| `debate` | Advocate, Critic, Moderator |
| `research` | Planner, Researcher, Verifier, Analyst, Evaluator, ReportWriter |
| `smart_assistant` | ToolSelector → ToolExecutor (deterministic) → Summarizer |
| `smart_travel_planner` | TravelToolSelector → ToolExecutor (deterministic) → TravelPlanner |
| `prompt_evaluation` | PromptParser → CriteriaJudge → Improver → EvalReporter (autogen only) |

**Shared `orchestrators/utils/`** — single source of truth, no duplication:
- `get_tool_registry()` — lazy map of 21 tool names → callables
- `resolve_tools(names)` — filter registry by requested names
- `build_tool_catalog(names)` — JSON-serialisable catalog for LLM selector prompts
- `execute_tool_calls(tool_calls, cache)` — parallel async execution with caching
- `extract_json_object(text)` — 4-stage JSON extraction (fast → markdown → generic → auto-repair)
- `normalize_tool_plan(raw, query, names)` — validate + clean LLM tool plan
- `normalize_travel_tool_plan(raw, query, names)` — travel-specific plan normalization
- `TRAVEL_TOOL_NAMES` — set of travel-only tool names (excludes web_search/scrape_url)
- `run_team(team, task)` — shared AutoGen stream runner (used by autogen + mcp)
- `build_executor_steps(results)` — convert tool result envelopes to step dicts
- `merge_steps(pre, post)` — renumber and concatenate step lists

**Adding a new workflow**: create `workflows/my_workflow.py` in both `autogen/workflows/` and `custom/workflows/`, add entry to `WORKFLOW_REGISTRY` in both orchestrators, add dispatcher method `_run_my_workflow`. Mirror in `crewai/orchestrator.py` + YAML configs if needed.

**Tool registry** (`tool_registry.py` → `get_tool_registry()`) — names match `agent_runner.REGISTRY`:

| Tool name | Wraps | Purpose |
|---|---|---|
| `web_search` | `tool_web_search` | DuckDuckGo / SerpAPI search |
| `scrape_url` | `tool_web_scraper` | Full page content extraction |
| `get_stock_price` | `tool_stock` | Real-time stock price via yfinance |
| `get_stock_history` | `tool_stock` | Historical stock prices |
| `generate_stock_chart` | `tool_chart` | Stock performance chart (matplotlib) |
| `get_crypto_price` | `tool_stock` | Real-time crypto price via yfinance |
| `generate_chart` | `tool_chart` | Generic chart from structured data |
| `get_weather` | `tool_weather` | Current weather conditions |
| `save_research_report` | `tool_file` | Structured markdown report + JSON sidecar |
| `search_flights` | `tool_travel` | Flight search (demo data) |
| `search_hotels` | `tool_travel` | Hotel search (demo data) |
| `estimate_trip_budget` | `tool_travel` | Trip budget estimation |
| `search_places` | `tool_travel` | Tourist attractions lookup |
| `search_restaurants` | `tool_travel` | Restaurant search |
| `generate_itinerary` | `tool_travel` | Day-wise itinerary generation |
| `get_local_transport_info` | `tool_travel` | Local transport options |
| `get_distance_between_places` | `tool_travel` | Distance/travel time between cities |
| `generate_trip_summary` | `tool_travel` | Trip highlights and tips |
| `get_currency_exchange` | `tool_travel` | Real currency conversion (exchangerate.host) |
| `get_geo_distance` | `tool_travel` | Real straight-line distance (OpenStreetMap) |

**`prompt_evaluation` workflow — 4-agent pipeline**:
1. **PromptParser** — extracts intent, variables, constraints, role framing, missing context
2. **CriteriaJudge** — scores on 5 criteria (Clarity, Specificity, Context, Safety, Token Efficiency 0-10) and lists concrete issues as JSON
3. **Improver** — rewrites the prompt fixing every issue while preserving original intent
4. **EvalReporter** — assembles final markdown report: scores table, issues list, improved prompt, changes made, verdict
- No tools required — pure LLM reasoning pipeline
- Input: raw prompt text as `question`
- Output: structured markdown evaluation report

**`smart_assistant` workflow — 3-agent pipeline**:
1. **ToolSelector** (LLM, max 2 steps) — analyses query, returns JSON tool plan with intent + args
2. **ToolExecutor** (deterministic, parallel) — runs selected tools via `_execute_tool_calls()` with caching
3. **Summarizer** (LLM) — synthesizes tool results into final answer

**`smart_travel_planner` workflow — 3-agent pipeline**:
1. **TravelToolSelector** (LLM, max 2 steps) — extracts travel entities (destination, days, budget, travelers, preferences), selects travel-only tools
2. **ToolExecutor** (deterministic, parallel) — runs selected travel tools
3. **TravelPlanner** (LLM) — formats structured travel plan (Overview, Budget, Hotels, Attractions, Weather, Transport, Tips)
- Travel tools are restricted to `_TRAVEL_TOOL_NAMES` set (excludes `web_search`/`scrape_url`)

**`research` workflow — 6-agent pipeline**:
- Planner → Researcher (with data tools) → Verifier → Analyst → Evaluator → ReportWriter (with `save_research_report`)
- ReportWriter saves final report as markdown + JSON sidecar to `user_uploaded_files/research_reports/`

- Tool names are **unified** with `agent_runner.REGISTRY` so `/tools` and `/tools/{name}/test` work for both orchestrators
- `run_team()` (`step_utils.py`) is the shared async stream runner used by all workflows
- Tool results are **cached** in `_tool_cache` passed from `AutoGenOrchestrator` (keyed by `tool_name:json(args)`)
- `build_executor_steps()` and `merge_steps()` (`step_utils.py`) are shared by smart_assistant and smart_travel_planner
- `_normalize_plan_base()` (`plan_normalizer.py`) is the shared core for both `normalize_tool_plan` and `normalize_travel_tool_plan`

**Tool Discovery — `/api/agents/tools`**:

All tools sourced from `orchestrators/utils/get_tool_registry()` — single source of truth. The old ITool-based `.tools` dict on `CustomOrchestrator` is removed; `tools.py` and `agent_runner.REGISTRY` are legacy only.

**Web Search Configuration**:
```bash
# Optional: upgrade from DuckDuckGo to SerpAPI
SERPAPI_KEY=your_serpapi_key  # If not set, uses DuckDuckGo (free)
```

**Dependencies**: `duckduckgo-search`, `beautifulsoup4`, `requests`, `yfinance`, `matplotlib` (optional for charts) (all in requirements.txt)

**Example — Multi-tool workflows**:
```bash
# Smart assistant (auto-selects tools)
curl -X POST "/api/agents/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Tesla stock price and weather in Austin?", "orchestrator_type": "autogen", "workflow": "smart_assistant"}'
# Flow: ToolSelector → [get_stock_price(TSLA), get_weather(Austin)] → Summarizer

# Smart travel planner
curl -X POST "/api/agents/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Plan a 3-day trip to Goa from Delhi with budget 25000 INR", "orchestrator_type": "autogen", "workflow": "smart_travel_planner"}'
# Flow: TravelToolSelector → [search_flights, search_hotels, estimate_trip_budget, search_places, generate_itinerary] → TravelPlanner

# Research with report saving
curl -X POST "/api/agents/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Research the impact of AI on healthcare", "orchestrator_type": "autogen", "workflow": "research"}'
# Flow: Planner → Researcher(web_search) → Verifier → Analyst → Evaluator → ReportWriter(save_research_report)
```

## 11. Paragraph-Aware Chunking System (NEW - Tier 1 Optimization)

### Overview

The system implements **cross-encoder reranking** as a Tier 1 optimization for retrieval quality. This is the single highest-impact improvement for RAG systems, fixing ordering mistakes from cosine similarity.

### Architecture

**Retrieval Pipeline**:
```
Query → Vector Search (top-20) → RBAC Filter → Cross-Encoder Rerank → Top-K to LLM
```

**Key Components**:
- **CrossEncoderReranker** (`app/modules/vector_db/reranker.py`)
- **Model**: `cross-encoder/ms-marco-MiniLM-L6-v2`
- **Integration**: RAG Orchestrator `retrieve_documents()` method

### Implementation Details

**Reranking Process**:
1. **Over-fetch**: Retrieve top-20 documents from vector store (4x requested or minimum 20)
2. **RBAC Filter**: Apply role-based access control
3. **Rerank**: Score query-document pairs with cross-encoder
4. **Return**: Top-K highest scoring documents to LLM

**Code Example**:
```python
from app.modules.vector_db.reranker import CrossEncoderReranker

reranker = CrossEncoderReranker()
reranked_docs = reranker.rerank(
    query="What is the vacation policy?",
    documents=retrieved_docs,  # 20 documents from vector search
    top_k=3  # Return top 3 after reranking
)
```

### Benefits

**Retrieval Quality Improvements**:
- ✅ **Better relevance**: Cross-encoder considers query-document interaction
- ✅ **Fixes cosine mistakes**: Corrects vector similarity ordering errors
- ✅ **Semantic understanding**: Captures nuanced relevance beyond embeddings
- ✅ **Higher precision**: More relevant documents reach the LLM

**Performance Characteristics**:
- **Latency**: ~100-200ms for 20 documents (CPU)
- **Memory**: ~400MB model size
- **Accuracy**: Significant improvement over vector similarity alone
- **Fallback**: Gracefully falls back to vector similarity on errors

### Configuration

**Model Selection**:
```python
# Default model (recommended)
reranker = CrossEncoderReranker()

# Custom model
reranker = CrossEncoderReranker(model_name="cross-encoder/ms-marco-TinyBERT-L-6")
```

**Retrieval Parameters**:
- **Over-fetch ratio**: 4x (retrieve 4x more than needed)
- **Minimum retrieval**: 20 documents
- **Final top-k**: Configurable per query (default: 3-5)

### Integration with RAG

**Automatic Reranking**:
- Enabled by default in `RAGOrchestrator.retrieve_documents()`
- No API changes required
- Transparent to existing queries

**Query Flow**:
```bash
curl -X POST "/api/rag/local/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the vacation policy?",
    "top_k": 3,  # Will retrieve 20, rerank, return top 3
    "use_llm": true
  }'
```

### Testing

**Test Script**: `test_reranker.py`
```bash
python test_reranker.py
```

**Expected Output**:
- Original top-3 by vector similarity
- Reranked top-3 by cross-encoder scores
- Comparison showing improved relevance

### Why This is Tier 1

**Highest ROI Optimization**:
1. **Minimal code changes**: Single module addition
2. **Maximum impact**: Directly improves what LLM sees
3. **No prompt engineering**: Fixes data quality at source
4. **Proven effectiveness**: Industry-standard approach
5. **Easy to validate**: Clear before/after metrics

**Comparison to Other Optimizations**:
- **Tier 1 (Reranking)**: Fixes retrieval → Better context → Better answers
- **Tier 2 (Prompt tuning)**: Works with what retrieval gives
- **Tier 3 (Model selection)**: Limited by input quality

### Model Information

**cross-encoder/ms-marco-MiniLM-L6-v2**:
- **Type**: Cross-encoder for passage ranking
- **Training**: MS MARCO passage ranking dataset
- **Size**: ~90M parameters, ~400MB
- **Speed**: ~5-10ms per query-document pair (CPU)
- **Accuracy**: State-of-art for passage reranking

### Future Enhancements

**Potential Improvements**:
- **Caching**: Cache reranking scores for repeated queries
- **Batch processing**: Parallel reranking for multiple queries
- **Model variants**: Support for larger/smaller models
- **Hybrid scoring**: Combine vector + rerank scores
- **A/B testing**: Compare reranked vs non-reranked results

## 11. Paragraph-Aware Chunking System (NEW - Tier 1 Optimization)

### Overview

The system implements **paragraph-aware chunking** that respects semantic boundaries instead of cutting text at arbitrary character positions. This keeps semantic units together for significantly better retrieval quality.

### The Problem with Fixed-Size Chunking

**Old Approach**:
- Cuts text every N characters
- Often splits mid-sentence or mid-paragraph
- Breaks semantic coherence
- Reduces retrieval quality

**Example of Bad Cut**:
```
Chunk 1: "...employees receive 20 days of vacation. Part-time employ"
Chunk 2: "ees receive vacation on a pro-rated basis..."
```

### Paragraph-Aware Solution

**New Approach**:
1. Split on double newlines (paragraph boundaries) first
2. Combine small paragraphs until reaching target size
3. Only split paragraphs if they exceed max size
4. Split at sentence boundaries when necessary
5. Add overlap between chunks

**Example of Good Cut**:
```
Chunk 1: "...employees receive 20 days of vacation."
Chunk 2: "Part-time employees receive vacation on a pro-rated basis..."
```

### Implementation

**Core Module**: `app/modules/core/chunking.py`

**Key Functions**:
```python
from app.modules.core.chunking import chunk_text_paragraph_aware, ChunkConfig

# Configure chunking
config = ChunkConfig(
    chunk_size=512,      # Target size
    min_chunk_size=100,  # Minimum size
    max_chunk_size=1024, # Maximum before forced split
    overlap=50           # Overlap between chunks
)

# Chunk with metadata
chunks_with_meta = chunk_text_paragraph_aware(text, config)

for chunk, meta in chunks_with_meta:
    print(f"Chunk: {len(chunk)} chars, {meta['paragraph_count']} paragraphs")
    print(f"Type: {meta['chunk_type']}")
```

**Backward Compatible**:
```python
# Existing code continues to work
from app.modules.core.utils import chunk_text_basic

chunks = chunk_text_basic(text, chunk_size=512, overlap=64)
# Now uses paragraph-aware chunking internally!
```

### Chunking Pipeline

**Step-by-Step Process**:

1. **Split into paragraphs**:
   ```python
   paragraphs = text.split('\n\n')
   ```

2. **Combine small paragraphs**:
   ```python
   # Combine until reaching target size
   while current_size < chunk_size:
       add_next_paragraph()
   ```

3. **Split large paragraphs**:
   ```python
   # If paragraph > max_size, split at sentences
   if len(paragraph) > max_size:
       split_at_sentence_boundaries()
   ```

4. **Add overlap**:
   ```python
   # Include last paragraph of previous chunk
   if overlap > 0:
       include_last_paragraph_for_context()
   ```

### Chunk Size Comparison

**Test Results** (from `test_chunking.py`):

| Chunk Size | Chunks Created | Avg Paragraphs | Quality |
|------------|----------------|----------------|----------|
| **256 chars** | More chunks | 1-2 | Good precision, may split paragraphs |
| **512 chars** | Balanced | 2-3 | **Recommended** - best balance |
| **1024 chars** | Fewer chunks | 4-6 | Better context, may be too large |

### Benefits

**Retrieval Quality**:
- ✅ **Semantic coherence**: Keeps related content together
- ✅ **No mid-sentence cuts**: Chunks end at natural boundaries
- ✅ **Better context**: Complete thoughts in each chunk
- ✅ **Improved relevance**: More meaningful retrieval results

**LLM Processing**:
- ✅ **Cleaner input**: No broken sentences
- ✅ **Better understanding**: Complete semantic units
- ✅ **Higher quality answers**: LLM sees coherent context

### Testing

**Test Script**: `test_chunking.py`
```bash
python test_chunking.py
```

**Output Includes**:
- Comparison of 256, 512, and 1024 character chunks
- Statistics (avg chars, tokens, paragraphs per chunk)
- Preview of each chunk
- Warnings for mid-sentence cuts
- Edge case testing

**Sample Output**:
```
CHUNK SIZE: 512 characters
============================================================
Total chunks created: 8

Chunk Statistics:
  - Average characters: 487.3
  - Min characters: 245
  - Max characters: 612
  - Average tokens (estimated): 121.8
  - Average paragraphs per chunk: 2.1

--- Chunk 1 ---
Characters: 487, Tokens: ~121, Paragraphs: 2
Type: paragraph_aware
Preview: # Company Vacation Policy

Our company values work-life balance...
✓ Chunk ends at sentence/paragraph boundary
```

### Configuration Options

**ChunkConfig Parameters**:
```python
config = ChunkConfig(
    chunk_size=512,      # Target size (recommended: 512)
    min_chunk_size=100,  # Minimum size (avoid tiny chunks)
    max_chunk_size=1024, # Maximum size (2x chunk_size recommended)
    overlap=50           # Overlap for context continuity
)
```

**Recommendations**:
- **General use**: 512 chars, 50 overlap
- **Long documents**: 1024 chars, 100 overlap
- **Short snippets**: 256 chars, 25 overlap

### Integration

**Automatic in Document Manager**:
- All document ingestion uses paragraph-aware chunking
- No code changes needed
- Existing documents can be re-chunked

**Manual Usage**:
```python
from app.modules.core.chunking import chunk_text_paragraph_aware_simple

chunks = chunk_text_paragraph_aware_simple(
    text=document_text,
    chunk_size=512,
    overlap=50
)
```

### Why This is Tier 1

**High Impact**:
1. **Better retrieval**: Semantic units improve relevance
2. **Better LLM input**: Clean, coherent chunks
3. **Easy to implement**: Drop-in replacement
4. **Immediate results**: Visible quality improvement
5. **No downsides**: Only benefits, no trade-offs

**Comparison**:
- **Fixed-size chunking**: Fast but breaks semantics
- **Paragraph-aware**: Slightly slower but much better quality
- **ROI**: High - minimal cost, significant benefit

### Edge Cases Handled

1. **Very long paragraphs**: Split at sentence boundaries
2. **Many small paragraphs**: Combine efficiently
3. **No paragraph breaks**: Fall back to sentence splitting
4. **Empty text**: Return empty list gracefully
5. **Single paragraph**: Return as single chunk if under max size

### Future Enhancements

- **Semantic similarity**: Group related paragraphs
- **Topic modeling**: Chunk by topic boundaries
- **Heading awareness**: Respect document structure
- **Language-specific**: Optimize for different languages
- **Adaptive sizing**: Adjust chunk size based on content type

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
    def __init__(self, model_client=None):
        # Use injected model client or create default
        self.model_client = model_client
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

### Configuration
```bash
# Optional cloud API keys (for cloud providers)
OPENAI_API_KEY=your_openai_key
GOOGLE_API_KEY=your_google_key
HUGGINGFACE_API_TOKEN=your_hf_token

# Custom/Third-party LLM provider (optional)
CUSTOMLLM_BASE_URL=http://localhost:8080
CUSTOMLLM_API_KEY=your_custom_api_key
# Legacy support (backward compatibility)
COLABLLM_BASE_URL=http://localhost:8080  # Falls back to CUSTOMLLM_BASE_URL
COLABLLM_API_KEY=your_api_key            # Falls back to CUSTOMLLM_API_KEY

# LlamaServer provider (NEW)
LLAMASERVER_BASE_URL=http://127.0.0.1:8080/v1
LLAMASERVER_MODEL_NAME=mistral-7b-instruct-v0.2

# CrewAI settings
CREW_BASE_URL=http://localhost:8080

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

### AutoGen Debug Logging

All AutoGen modules emit `logger.debug(...)` at key checkpoints. Enable with:
```python
import logging
logging.getLogger("app.modules.agents").setLevel(logging.DEBUG)
```

| Module | What is logged |
|---|---|
| `autogen_orchestrator.py` | Workflow dispatch (name, tools, max_steps), completion (steps, tools_used, answer_len) |
| `tool_registry.py` | Registry build start, registered tool count + names (logged once, lazy) |
| `tool_utils.py` | `resolve_tools` input/output, each tool START/cache-HIT/DONE/FAILED, parallel batch start/done |
| `json_utils.py` | Which parse path succeeded (fast/markdown/generic/auto-repair), no-match |
| `plan_normalizer.py` | Raw call count, each tool accepted/skipped with reason, normalized count, fallback triggers, intent+confidence |
| `step_utils.py` | Task start (first 120 chars), each message (agent, step, tool_calls or content_len), final summary |
| `workflows/debate.py` | Start (query_len, tools, max_steps), done (steps, tools_used, answer_len) |
| `workflows/research.py` | Start, done |
| `workflows/smart_assistant.py` | Start, selector result (intent/confidence/routing), executor result count, done |
| `workflows/smart_travel_planner.py` | Start, selector result (intent/confidence/destination/routing), executor result count, done |
| `workflows/prompt_evaluation.py` | Start (query_len, max_steps), done (steps, answer_len) |

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

# Alternative OpenAI endpoint
curl -X POST "/api/rag/openai/query" \
  -H "Authorization: Bearer <token>" \
  -d '{"question": "What is our policy?", "use_llm": true}'

# Hugging Face
curl -X POST "/api/rag/huggingface/query" \
  -H "Authorization: Bearer <token>" \
  -d '{"question": "What is our policy?", "use_llm": true}'

# Alternative Hugging Face endpoint
curl -X POST "/api/rag/hf/query" \
  -H "Authorization: Bearer <token>" \
  -d '{"question": "What is our policy?", "use_llm": true}'

# CustomLLM (preferred for third-party APIs)
curl -X POST "/api/rag/customllm/query" \
  -H "Authorization: Bearer <token>" \
  -d '{"question": "What is our policy?", "use_llm": true}'

# ColabLLM (legacy - backward compatibility)
curl -X POST "/api/rag/colabllm/query" \
  -H "Authorization: Bearer <token>" \
  -d '{"question": "What is our policy?", "use_llm": true}'

# LlamaServer (NEW)
curl -X POST "/api/rag/llamaserver/query" \
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
# Query (auto-creates conversation if no conversation_id given)
curl -X POST "/api/agents/query" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the status of my tickets?", "max_steps": 3}'
# Returns conversation_id — use it for follow-ups and history

# Continue existing conversation
curl -X POST "/api/agents/query" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "Any updates?", "conversation_id": "conv_xxx"}'

# Get agent conversation history (from agent_messages table)
curl -X GET "/api/agents/conversations/conv_xxx/messages" \
  -H "Authorization: Bearer <token>"

# Test a tool (JSON body, not query param)
curl -X POST "/api/agents/tools/web_search/test" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"input_data": "latest AI news 2025"}'

# Test stock tool
curl -X POST "/api/agents/tools/get_stock_price/test" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"input_data": "AAPL"}'

# Test multi-arg tool (save_text_file)
curl -X POST "/api/agents/tools/save_text_file/test" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"input_data": "{\"filename\": \"out.txt\", \"content\": \"hello world\"}"}"

# List all available tools
curl -X GET "/api/agents/tools" -H "Authorization: Bearer <token>"

# List AutoGen workflows and tools
curl -X GET "/api/agents/autogen/workflows" -H "Authorization: Bearer <token>"

# AutoGen debate workflow with specific tools
curl -X POST "/api/agents/query" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "Should AI replace human jobs?", "orchestrator_type": "autogen", "workflow": "debate", "tools": ["web_search", "get_stock_price"]}'

# AutoGen research workflow with all tools
curl -X POST "/api/agents/query" \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"question": "Research Tesla stock and Austin weather", "orchestrator_type": "autogen", "workflow": "research"}'
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

**Last Updated**: 2025-06-01 (Corrected CrewAI module location to `orchestrators/crewai/`; added `mcp_client_stdio.py`, `audio_utils.py`, `colabllm_rag_service.py`; unified CrewAI under `/api/agents/query`; added `data/globalCompany/` and `data/cleaned/`; added `requirements_mcp.txt`; corrected `prompt_evaluation` workflow: 4-agent pipeline PromptParser→CriteriaJudge→Improver→EvalReporter; scores on 5 criteria; produces structured markdown report with improved prompt and verdict)

**Recent changes (autogen/custom/mcp refactor)**:
- `orchestrators/utils/` created as shared package — `tool_registry`, `tool_utils`, `json_utils`, `plan_normalizer`, `step_utils` moved here; all three orchestrators import from this single location, zero duplication
- `orchestrators/autogen/` — removed local utility copies; workflows now import from `...utils`
- `orchestrators/custom/` — fully rewritten: `CustomOrchestrator` now mirrors `AutoGenOrchestrator` with same 4 workflows (`debate`, `research`, `smart_assistant`, `smart_travel_planner`) and same `WORKFLOW_REGISTRY` pattern; uses `llm_fn(system, user) → str` instead of AutoGen agents; `_echo_llm` fallback for testing
- `orchestrators/mcp/` — added to `orchestrators/__init__.py` exports; `MCPOrchestrator` + `MCPClient` now available via `AgentOrchestratorFactory`
- `factories.py` — added `mcp` orchestrator type; `custom` now receives `llm_fn` wired from `LlamaServerProvider`; `get_available_types()` returns `{custom, autogen, mcp}`
- `api_routes_agents.py` — updated: `orchestrator_type` default `autogen`; `workflow` default `smart_assistant`; `/workflows` endpoint replaces `/autogen/workflows` (supports all 3 via `?orchestrator_type=`); `/status` returns `orchestrator_types` map; tool lookup uses shared registry only; `AgentQueryResponse` adds `available_workflows` and `orchestrator_type` fields

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