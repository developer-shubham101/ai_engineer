# Project Structure

## Directory Layout

```
ai_backend/
├── app/                        # Main application package
│   ├── modules/                # Modular service architecture
│   │   ├── agents/             # AutoGen agent orchestration
│   │   │   ├── function_tools/ # Tool implementations (web, RAG, finance)
│   │   │   ├── orchestrators/  # Agent workflow orchestrators
│   │   │   ├── agent_runner.py # Execution runner
│   │   │   ├── factories.py    # AgentOrchestratorFactory
│   │   │   ├── interfaces.py   # IAgentOrchestrator interface
│   │   │   └── tools.py        # Tool definitions
│   │   ├── api/                # Shared API utilities
│   │   │   ├── handlers.py     # Common request handlers
│   │   │   ├── models.py       # Shared Pydantic models
│   │   │   └── validators.py   # Input validation helpers
│   │   ├── auth/               # Authentication & authorization
│   │   │   ├── interfaces.py   # IAuthenticator, IRBACManager
│   │   │   ├── jwt_auth.py     # JWTAuthenticator implementation
│   │   │   ├── rbac.py         # RBAC logic and document filtering
│   │   │   ├── session_manager.py  # SQLiteSessionManager
│   │   │   └── user_manager.py     # SQLiteUserManager
│   │   ├── config/             # Configuration and constants
│   │   │   ├── constants.py    # Enums, role levels, defaults, API prefixes
│   │   │   ├── database_config.py  # DB file names and paths
│   │   │   ├── settings.py     # Settings class (env vars, paths)
│   │   │   ├── local_models.json       # Local GGUF model registry
│   │   │   ├── multimodal_models.json  # Multimodal model registry
│   │   │   └── onboarding_fields.json  # Onboarding config
│   │   ├── conversation/       # Conversation history management
│   │   │   └── conversation_manager.py  # SQLiteConversationManager
│   │   ├── core/               # Core document operations
│   │   │   ├── chunking.py         # Text chunking strategies
│   │   │   ├── cleanup_service.py  # Document cleanup utilities
│   │   │   ├── document_manager.py # DocumentManager (add/search/delete)
│   │   │   ├── metadata_generator.py  # LLM-based metadata generation
│   │   │   ├── metadata_models.py  # Pydantic metadata schemas
│   │   │   ├── profile_analyzer.py # User/document profile analysis
│   │   │   ├── utils.py            # Core utility functions
│   │   │   └── version_manager.py  # Document versioning (SQLite)
│   │   ├── crew_ai/            # CrewAI multi-agent workflows
│   │   │   ├── custom_llm.py   # CrewAI-compatible LLM wrapper
│   │   │   ├── factory.py      # CrewOrchestratorFactory
│   │   │   ├── interfaces.py   # ICrewOrchestrator
│   │   │   ├── orchestrator.py # Debate/research workflow logic
│   │   │   └── travel_workflow.py  # Example travel planning crew
│   │   ├── llm/                # LLM provider layer
│   │   │   ├── providers/      # Provider implementations
│   │   │   │   ├── local.py    # LocalLLMProvider (llama-cpp)
│   │   │   │   ├── google.py   # GoogleLLMProvider (Gemini)
│   │   │   │   ├── gpt.py      # GPTLLMProvider (OpenAI)
│   │   │   │   └── huggingface.py  # HuggingFaceLLMProvider
│   │   │   ├── services/       # Higher-level LLM services
│   │   │   ├── prompt_templates/   # LangChain prompt templates
│   │   │   ├── interfaces.py   # ILLMProvider interface
│   │   │   ├── model_manager.py    # Local model loading/caching
│   │   │   ├── prompt_builder.py   # Token-aware prompt construction
│   │   │   ├── prompt_chain.py     # LangChain chain integration
│   │   │   ├── prompt_manager.py   # Prompt lifecycle management
│   │   │   ├── provider_factory.py # LLM provider factory
│   │   │   ├── rag_orchestrator.py # Main RAG query pipeline
│   │   │   ├── template_manager.py # Prompt template CRUD
│   │   │   ├── colabllm_plugin.py  # ColabLLM/CustomLLM plugin
│   │   │   ├── llamaserver_plugin.py  # LlamaServer plugin
│   │   │   └── middleware.py   # LLM request middleware
│   │   ├── multimodal/         # Audio, vision, emotion processing
│   │   │   ├── audio_utils.py      # Audio preprocessing
│   │   │   ├── emotion_providers.py # Emotion detection
│   │   │   ├── file_manager.py     # Uploaded file management
│   │   │   ├── interfaces.py       # ISTTProvider, ITTSProvider, IVisionProvider
│   │   │   ├── stt_providers.py    # Vosk, Whisper STT
│   │   │   ├── tts_providers.py    # pyttsx3 TTS
│   │   │   └── vision_providers.py # OCR, YOLO vision
│   │   ├── vector_db/          # Vector store and retrieval
│   │   │   ├── bm25_index.py       # BM25 keyword index
│   │   │   ├── chroma_impl.py      # ChromaVectorStore
│   │   │   ├── embedding_manager.py # Sentence-transformer embeddings
│   │   │   ├── faiss_vector_store.py # FaissVectorStore
│   │   │   ├── hybrid_retrieval.py  # BM25 + vector fusion
│   │   │   ├── interfaces.py        # IVectorStore interface
│   │   │   ├── query_preprocessor.py # Query cleaning/expansion
│   │   │   └── reranker.py          # CrossEncoderReranker
│   │   ├── integration.py      # DI Container (get_container())
│   │   └── README.md
│   ├── utils/
│   │   └── doc_parser.py       # Multi-format document parser
│   ├── api_routes_agents.py    # /api/agents/* endpoints
│   ├── api_routes_audio.py     # /api/audio/* endpoints
│   ├── api_routes_auth.py      # /api/auth/* endpoints
│   ├── api_routes_cleanup.py   # /api/cleanup/* endpoints
│   ├── api_routes_conversations.py  # /api/conversations/* endpoints
│   ├── api_routes_crew.py      # /api/crew/* endpoints
│   ├── api_routes_media.py     # /api/media/* endpoints
│   ├── api_routes_models.py    # /api/models/* endpoints
│   ├── api_routes_rag.py       # /api/rag/* endpoints (main RAG)
│   ├── api_routes_templates.py # /api/templates/* endpoints
│   ├── api_routes_vision.py    # /api/vision/* endpoints
│   ├── dependencies.py         # FastAPI dependency providers
│   ├── logging_config.py       # Structured logging setup
│   └── main.py                 # FastAPI app, lifespan, router registration
├── data/                       # Sample and training documents
│   ├── company/v1/, v2/        # Versioned company documents with .meta.json
│   ├── companyData/            # Flat company document collection
│   ├── examples/               # Example documents by department
│   ├── globalCompany/v1/       # Financial PDFs (AAPL, AMZN, MSFT, etc.)
│   └── missions_output/        # Space mission text files
├── scripts/                    # Utility and setup scripts
│   ├── train_sentiment.py      # Sentiment model training
│   ├── seed_examples.py        # Database seeding
│   ├── download_*.py           # Model download helpers
│   └── test_*.py               # Script-level tests
├── test_module/                # Integration test suite
│   ├── conftest.py             # Pytest fixtures
│   ├── test_*.py               # Test files per module
│   └── pytest.ini
├── crew_config/                # CrewAI YAML configuration
│   ├── agents.yaml             # Agent role definitions
│   └── tasks.yaml              # Task definitions
├── documents/                  # Developer documentation
├── archive/                    # Historical docs and planning notes
├── models/                     # Local GGUF model files (gitignored)
├── database/                   # SQLite databases (gitignored)
├── chroma_storage/             # ChromaDB persistence (gitignored)
├── sentiment/                  # Trained sentiment model artifacts
├── .env.example                # Environment variable template
├── docker-compose.yml          # Docker deployment
├── Dockerfile
├── requirements.txt            # Core dependencies
├── requirements_agents.txt     # Agent-specific deps
├── requirements_multimodal.txt # Multimodal deps
├── requirements_pdf.txt        # PDF parsing deps
└── run_app.py                  # Entry point
```

## Core Components & Relationships

### Dependency Injection Container (`app/modules/integration.py`)
Central `Container` class wires all services together. Accessed globally via `get_container()`. Services are initialized once and reused (singleton pattern). Supports `override_instance()` for test mocking.

```
Container
├── SQLiteUserManager
├── SQLiteSessionManager
├── EmbeddingManager
├── FaissVectorStore | ChromaVectorStore  (env-driven)
├── VersionManager
├── DocumentManager  ──────────────────── depends on VectorStore + VersionManager
├── JWTAuthenticator ──────────────────── depends on UserManager
├── RAGOrchestrator  ──────────────────── depends on VectorStore + SessionManager
├── SQLiteConversationManager  (lazy)
├── TemplateManager            (lazy)
├── AgentOrchestrator          (lazy)
├── CrewOrchestrator           (lazy)
├── CrossEncoderReranker       (lazy)
├── BM25Index                  (lazy)
└── LLMMetadataGenerator       (lazy)
```

### Request Flow
```
HTTP Request
  → FastAPI Router (api_routes_*.py)
  → Depends(get_current_user) [JWT validation]
  → Depends(require_roles([...])) [RBAC check]
  → get_container().get_rag_orchestrator()
  → RAGOrchestrator.query()
      → QueryPreprocessor (spell check, expansion)
      → VectorStore.search() + BM25Index (hybrid)
      → CrossEncoderReranker.rerank()
      → RBAC filter (sensitivity/department check)
      → PromptBuilder (token budgeting)
      → LLMProvider.generate()
  → Response
```

## Architectural Patterns

- **Dependency Injection**: Container pattern, not framework DI
- **Interface Segregation**: All major services have `interfaces.py` with abstract base classes
- **Factory Pattern**: `AgentOrchestratorFactory`, `CrewOrchestratorFactory`, `provider_factory.py`
- **Plugin Pattern**: LLM providers as interchangeable plugins
- **Repository Pattern**: `DocumentManager`, `VersionManager` abstract storage details
- **Singleton Services**: Container manages single instances per service type
- **Lazy Initialization**: Expensive services (reranker, BM25, agents) initialized on first use
