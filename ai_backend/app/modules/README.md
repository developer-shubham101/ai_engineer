# 🏗️ Modular Architecture

This directory contains the new modular architecture for the AI Backend system, designed for maintainability, testability, and scalability.

## 📋 Overview

The modular architecture separates the system into distinct, loosely-coupled modules with clear responsibilities:

1. **API Module** - Request/response handling and validation
2. **Config Module** - Configuration management and settings
3. **Auth Module** - Authentication, sessions, and RBAC
4. **Vector DB Module** - Document storage and retrieval
5. **LLM Module** - Language model providers and RAG orchestration
6. **Core Module** - Business logic and utilities

## 🎯 Design Principles

### 1. Interface-Based Design
Every module defines clear interfaces (abstract base classes) that specify contracts without implementation details.

```python
from app.modules.vector_db.interfaces import IVectorStore

class MyCustomVectorStore(IVectorStore):
    async def add_document(self, text: str, metadata: dict) -> str:
        # Custom implementation
        pass
```

### 2. Dependency Injection
All dependencies are managed through a central container, making the system highly testable and configurable.

```python
from app.modules.integration import get_container

container = get_container()
vector_store = container.get_vector_store()  # Gets configured implementation
```

### 3. Separation of Concerns
Each module has a single, well-defined responsibility:

- **API**: Only handles HTTP requests/responses
- **Auth**: Only handles authentication and authorization
- **Vector DB**: Only handles document storage and search
- **LLM**: Only handles language model interactions
- **Core**: Only handles business logic

## 📁 Module Structure

```
modules/
├── api/                    # API Layer
│   ├── models.py          # Pydantic request/response models
│   ├── handlers.py        # Request processing logic
│   └── validators.py      # Input validation
├── config/                # Configuration
│   ├── settings.py        # Environment and app settings
│   ├── constants.py       # Constants and enums
│   └── models.py          # Configuration data models
├── auth/                  # Authentication & Authorization
│   ├── interfaces.py      # Auth interfaces
│   ├── jwt_auth.py        # JWT implementation
│   ├── user_manager.py    # User management
│   ├── session_manager.py # Session handling
│   └── rbac.py           # Role-based access control
├── vector_db/            # Vector Database
│   ├── interfaces.py      # Vector DB interfaces
│   ├── chroma_impl.py     # ChromaDB implementation
│   └── embedding_manager.py # Embedding providers
├── llm/                  # Language Models
│   ├── interfaces.py      # LLM interfaces
│   ├── providers.py       # LLM implementations
│   ├── rag_orchestrator.py # RAG workflow
│   └── prompt_manager.py  # Prompt engineering
├── core/                 # Business Logic
│   ├── document_manager.py # Document operations
│   ├── version_manager.py  # Version control
│   ├── profile_analyzer.py # User profiling
│   └── utils.py           # Utility functions
└── integration.py        # Dependency injection container
```

## 🚀 Quick Start

### Basic Usage

```python
from app.modules.integration import get_container

# Initialize the container
container = get_container()
container.initialize()

# Get any service
user_manager = container.get_user_manager()
vector_store = container.get_vector_store()
rag_orchestrator = container.get_rag_orchestrator()
```

### Processing a RAG Query

```python
from app.modules.llm.interfaces import RAGRequest

# Create request
request = RAGRequest(
    question="What are our company policies?",
    user={"user_id": "123", "role": "Employee", "department": "HR"},
    top_k=5,
    use_llm=True
)

# Process query
orchestrator = container.get_rag_orchestrator()
response = await orchestrator.process_query(request)

print(f"Answer: {response.answer}")
print(f"Retrieved {len(response.retrieved_documents)} documents")
```

### Adding Documents

```python
# Get document manager
doc_manager = container.get_document_manager()

# Add document
doc_id = await doc_manager.add_document(
    text="This is a company policy document...",
    metadata={
        "source": "HR Manual",
        "department": "HR",
        "sensitivity": "department_confidential"
    },
    user={"user_id": "123", "role": "HR", "department": "HR"}
)
```

## 🧪 Testing

### Unit Testing with Mocks

```python
from unittest.mock import Mock
from app.modules.vector_db.interfaces import IVectorStore

# Create mock
mock_vector_store = Mock(spec=IVectorStore)
mock_vector_store.search_documents.return_value = []

# Inject mock
container._instances["vector_store"] = mock_vector_store

# Test your code
result = await some_function_that_uses_vector_store()
```

### Integration Testing

```python
# Use real implementations for integration tests
container = get_container()
container.initialize()

# Test end-to-end workflows
response = await container.get_rag_orchestrator().process_query(request)
assert response.answer is not None
```

## 🔄 Swapping Implementations

### Change Vector Database

```python
from app.modules.vector_db.pinecone_impl import PineconeVectorStore

# Swap ChromaDB for Pinecone
container._instances["vector_store"] = PineconeVectorStore(api_key="...")
```

### Change LLM Provider

```python
from app.modules.llm.providers import OpenAIProvider

# Add new provider
provider = OpenAIProvider(api_key="...")
orchestrator = container.get_rag_orchestrator()
orchestrator.register_provider("openai", provider)
```

### Change Authentication

```python
from app.modules.auth.oauth_auth import OAuthAuthenticator

# Swap JWT for OAuth
container._instances["authenticator"] = OAuthAuthenticator(...)
```

## 📊 Benefits

### ✅ Maintainability
- Clear module boundaries
- Single responsibility principle
- Easy to understand and modify

### ✅ Testability
- Mock any component easily
- Test modules in isolation
- Clear test boundaries

### ✅ Scalability
- Add new implementations without changing existing code
- Horizontal scaling of individual modules
- Easy to optimize specific components

### ✅ Flexibility
- Swap implementations at runtime
- Support multiple providers simultaneously
- Easy configuration management

## 🔧 Extending the Architecture

### Adding a New Vector Database

1. **Create Implementation**:
```python
# app/modules/vector_db/pinecone_impl.py
from .interfaces import IVectorStore

class PineconeVectorStore(IVectorStore):
    async def add_document(self, text: str, metadata: dict) -> str:
        # Pinecone-specific implementation
        pass
```

2. **Register in Container**:
```python
# In integration.py
container._instances["vector_store"] = PineconeVectorStore()
```

### Adding a New LLM Provider

1. **Create Provider**:
```python
# app/modules/llm/anthropic_provider.py
from .interfaces import ILLMProvider

class AnthropicProvider(ILLMProvider):
    async def generate(self, prompt: str, **kwargs) -> LLMResponse:
        # Anthropic-specific implementation
        pass
```

2. **Register Provider**:
```python
orchestrator = container.get_rag_orchestrator()
orchestrator.register_provider("anthropic", AnthropicProvider())
```

## 📚 Migration Guide

The system is gradually migrating from the legacy `services/` directory to this modular architecture:

1. **Legacy services** are still functional
2. **New features** should use the modular architecture
3. **Existing code** will be migrated incrementally
4. **API compatibility** is maintained during transition

## 🤝 Contributing

When adding new features:

1. **Define interfaces first** - Create abstract base classes
2. **Implement concrete classes** - Follow existing patterns
3. **Add to container** - Register in `integration.py`
4. **Write tests** - Both unit and integration tests
5. **Update documentation** - Keep this README current

## 📖 Further Reading

- [APP_CONTEXT.md](../../APP_CONTEXT.md) - Complete system documentation
- [test_modular_architecture.py](test_module/test_modular_architecture.py) - Example usage
- Individual module README files (coming soon)