# Project Structure & Architecture

## Directory Organization

### Core Application (`app/`)
Main application code with modular architecture:

- **`modules/`** - Core business logic modules
  - **`agents/`** - AI agent orchestration and tools
  - **`api/`** - API handlers, models, and validators
  - **`auth/`** - Authentication, RBAC, session management
  - **`config/`** - Configuration management and model definitions
  - **`conversation/`** - Conversation and session handling
  - **`core/`** - Document management, versioning, utilities
  - **`llm/`** - LLM providers, prompt management, RAG orchestration
  - **`multimodal/`** - Audio, vision, emotion processing
  - **`vector_db/`** - Vector database implementations (ChromaDB, FAISS)

- **`utils/`** - Utility functions and document parsing
- **`api_routes_*.py`** - FastAPI route definitions by feature
- **`main.py`** - Application entry point and FastAPI setup
- **`dependencies.py`** - Dependency injection configuration

### Data & Storage
- **`data/`** - Sample documents with versioning structure
  - **`company/`** - Versioned company documents (v1, v2)
  - **`examples/`** - Example documents for testing
  - **`missions_output/`** - Generated mission data
- **`database/`** - SQLite databases for users, conversations, sessions
- **`chroma_data/`** - ChromaDB vector storage
- **`models/`** - Local LLM models (GGUF format) and embeddings
- **`embeddings_models/`** - Sentence transformer models

### Development & Testing
- **`tests/`** - Comprehensive test suite
- **`test_module/`** - Modular testing framework
- **`scripts/`** - Utility scripts for setup, training, testing
- **`archive/`** - Historical documentation and deprecated code
- **`documents/`** - Project documentation and guides

### Configuration & Deployment
- **`.amazonq/rules/memory-bank/`** - AI assistant memory bank
- **`logs/`** - Application logging output
- **`user_uploaded_files/`** - User file uploads by user ID
- **`sample_response_query/`** - API response examples

## Core Components & Relationships

### LLM Provider Architecture
```
Provider Factory
├── Local Provider (llama-cpp-python)
├── OpenAI Provider (openai)
├── Google Provider (google-generativeai)
└── Hugging Face Provider (transformers)
```

### RAG Orchestration Flow
```
API Request → RAG Orchestrator → Provider Service → Base RAG Service
                    ↓
Vector DB ← Document Manager ← RBAC Filter ← Session Manager
```

### Authentication & Authorization
```
JWT Auth → User Manager → RBAC System → Document Filter
    ↓
Session Manager → Conversation Manager → Audit Logger
```

### Vector Database Layer
```
Embedding Manager → Vector Store Interface
                        ├── ChromaDB Implementation
                        └── FAISS Implementation
```

## Architectural Patterns

### Dependency Injection
- **Provider Factory Pattern**: Dynamic LLM provider selection
- **Interface Segregation**: Clean abstractions for vector stores, auth
- **Service Layer**: Business logic separated from API routes

### Modular Design
- **Feature-Based Modules**: Each module handles specific domain
- **Plugin Architecture**: Easy addition of new providers/features
- **Configuration-Driven**: Behavior controlled via settings files

### Security Architecture
- **Layered Security**: Authentication → Authorization → Data Filtering
- **Metadata-Based Access**: Document-level security controls
- **Audit Trail**: Comprehensive logging for compliance

### Data Management
- **Document Versioning**: Non-destructive updates with history
- **Session Persistence**: Conversation state management
- **Caching Strategy**: Embedding and model caching for performance

## Key Design Principles

### Offline-First
- Local model support for complete offline operation
- Graceful degradation when cloud services unavailable
- Local vector database storage

### Enterprise-Ready
- Role-based access control with flexible overrides
- Comprehensive audit logging
- Production-grade error handling and monitoring

### Extensibility
- Plugin architecture for new LLM providers
- Configurable prompt templates
- Modular vector database backends

### Performance Optimization
- Smart token budgeting and context truncation
- Efficient embedding caching
- Optimized prompt compression

## Integration Points

### External Services
- **OpenAI API**: GPT model integration
- **Google AI**: Gemini model integration  
- **Hugging Face**: Model hub integration
- **Local Models**: llama-cpp-python for GGUF models

### Storage Systems
- **ChromaDB**: Primary vector database
- **FAISS**: Alternative vector storage
- **SQLite**: User data, sessions, conversations
- **File System**: Document storage and model files

### Development Tools
- **FastAPI**: Web framework with automatic OpenAPI docs
- **Pytest**: Comprehensive testing framework
- **Docker**: Containerization support
- **Logging**: Structured logging with multiple levels