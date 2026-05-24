# Development Guidelines & Standards

## Code Quality Standards

### Import Organization
- `from __future__ import annotations` at the top of every module for forward references
- Standard library → third-party → local imports, each group separated by a blank line
- Local imports use absolute paths (`from app.modules.config.constants import ...`)
- Conditional/lazy imports inside functions for optional or heavy dependencies:
  ```python
  def get_reranker(self):
      if "reranker" not in self._instances:
          from .vector_db.reranker import CrossEncoderReranker
          self._instances["reranker"] = CrossEncoderReranker()
  ```
- `__all__` lists defined in every `__init__.py` to control public API surface

### Type Annotations
- All function parameters and return types annotated
- Use `Dict[str, Any]`, `List[Dict[str, Any]]`, `Optional[str]` from `typing`
- Interface types in signatures, concrete types in implementations
- Type hints on local variables when non-obvious: `document_manager: DocumentManager = ...`

### Naming Conventions
- Classes: `PascalCase` (e.g., `RAGOrchestrator`, `SQLiteUserManager`)
- Functions/methods: `snake_case`
- Constants: `UPPER_SNAKE_CASE` (e.g., `DEFAULT_TOP_K`, `VALID_SENSITIVITY_LEVELS`)
- Private container keys: lowercase strings (`"user_manager"`, `"vector_store"`)
- Route files: `api_routes_<domain>.py`
- Interface files: `interfaces.py` per module
- Enum values: string literals matching their semantic meaning (`"SuperAdmin"`, `"public_internal"`)

### Docstrings
- Module-level docstrings: one-line description in triple quotes
- Function docstrings: description + Args + Returns sections for public APIs
- Inline comments for non-obvious logic; avoid redundant comments

---

## Architectural Patterns

### Dependency Injection Container
All services accessed through the global `Container` via `get_container()`. Never instantiate services directly in route handlers.

```python
# Correct pattern
container = get_container()
container.initialize()
service = container.get_rag_orchestrator()

# Wrong — do not do this in routes
orchestrator = RAGOrchestrator(...)
```

`container.initialize()` is idempotent — safe to call multiple times (guarded by `_initialized` flag).

### Interface Segregation
Every module has an `interfaces.py` defining abstract base classes. Depend on interfaces, not implementations:
- `ILLMProvider` — all LLM providers implement this
- `IVectorStore` — ChromaVectorStore and FaissVectorStore implement this
- `IAgentOrchestrator`, `ICrewOrchestrator`, `IConversationManager`, etc.

### Factory Pattern
Use factory classes for provider/orchestrator creation:
```python
# LLM providers
provider = LLMProviderFactory.create(provider_name)

# Agents
orchestrator = AgentOrchestratorFactory.create_orchestrator(vector_store=vs)

# CrewAI
crew = CrewOrchestratorFactory.create_orchestrator()

# Multimodal
stt = create_stt_provider(provider_name)
```

### Lazy Initialization
Expensive services (reranker, BM25, agents, metadata generator) are initialized on first access, not at startup:
```python
def get_reranker(self):
    if "reranker" not in self._instances:
        from .vector_db.reranker import CrossEncoderReranker
        self._instances["reranker"] = CrossEncoderReranker()
    return self._instances.get("reranker")
```

### Module `__init__.py` Pattern
Each module's `__init__.py` exports its public interface and implementation:
```python
"""Module description."""
from .interfaces import IInterface
from .implementation import ConcreteImpl
from .factory import create_provider

__all__ = ["IInterface", "ConcreteImpl", "create_provider"]
```

---

## API Design Standards

### Router Setup
```python
from app.modules.config.constants import RAG_PREFIX
router = APIRouter(prefix=RAG_PREFIX, tags=["RAG"])
```
Always use constants for prefixes, never hardcode strings.

### Authentication Dependencies
```python
# Required auth
requester: Dict[str, Any] = Depends(get_current_user)

# Optional auth (Guest fallback)
requester: Optional[Dict[str, Any]] = Depends(get_current_user_optional)

# Role-gated endpoint (as dependency, not parameter)
@router.post("/admin", dependencies=[Depends(require_roles(SUPER_ADMIN_ROLES))])
```

### Request/Response Models
- All request bodies: Pydantic `BaseModel` with `Field()` defaults from constants
- All responses: typed Pydantic `BaseModel` with `response_model=` on the decorator
- Use `Field(default_factory=list)` for list fields, not mutable defaults

```python
class QueryRequest(BaseModel):
    question: str
    top_k: int = DEFAULT_TOP_K
    max_tokens: int = DEFAULT_MAX_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    conversation_id: Optional[str] = None

class QueryResponse(BaseModel):
    answer: Optional[str] = None
    retrieved: List[RetrievedDoc] = Field(default_factory=list)
```

### Error Handling Pattern
```python
try:
    result = await service.operation()
    return SuccessResponse(data=result)
except HTTPException:
    raise  # Re-raise HTTP exceptions unchanged
except ValueError as e:
    raise HTTPException(status_code=404, detail=str(e))
except Exception as e:
    logger.exception("Operation failed: %s", e)
    raise HTTPException(status_code=500, detail=str(e))
```

---

## Security & RBAC

### Role Constants (use these, never hardcode strings)
```python
from app.modules.config.constants import (
    SUPER_ADMIN_ROLES,      # ["SuperAdmin"]
    MANAGER_PLUS_ROLES,     # ["SuperAdmin", "Manager"]
    HR_PLUS_ROLES,          # ["SuperAdmin", "Manager", "HR"]
    EMPLOYEE_PLUS_ROLES,    # ["SuperAdmin", "Manager", "HR", "Employee"]
    ROLE_LEVELS,            # {"SuperAdmin": 4, "Manager": 3, ...}
    HR_LEVEL_THRESHOLD,     # 2
)
```

### Metadata Validation
Always validate sensitivity and department against constants before persisting:
```python
from app.modules.config.constants import VALID_SENSITIVITY_LEVELS, VALID_DEPARTMENTS

if sens and sens not in VALID_SENSITIVITY_LEVELS:
    raise HTTPException(status_code=400, detail=f"Invalid sensitivity '{sens}'...")
```

### Security Logging
Use structured logging helpers — never use raw `logger.info` for security events:
```python
from app.logging_config import log_security_event, log_user_action

log_security_event(logger, "ACCESS_DENIED", user_id,
                   role=user_role, resource=resource_id)

log_user_action(logger, "DOCUMENT_CREATED", user_id,
                document_id=doc_id, sensitivity=sensitivity,
                chunk_count=count, version=version)
```

### Department Ownership Check Pattern
```python
user_level = ROLE_LEVELS.get(user_role, 0)
if user_level < HR_LEVEL_THRESHOLD:
    if current_dept != user_dept:
        log_security_event(logger, "RBAC_UPDATE_DENIED", ...)
        raise HTTPException(status_code=403, detail=...)
```

---

## Configuration & Constants

### Settings Access
```python
from app.modules.config.settings import settings

db_path = settings.DATABASE_DIR / settings.CONVERSATIONS_DB_NAME
models_dir = settings.MODELS_DIR
```

### Constants Access
```python
from app.modules.config.constants import (
    DEFAULT_TOP_K, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE,
    DEFAULT_SENSITIVITY, DEFAULT_DEPARTMENT,
    VALID_PROVIDERS, VALID_SENSITIVITY_LEVELS, VALID_DEPARTMENTS
)
```

### Enums
Use enums for type-safe values; use `.value` when passing to external APIs:
```python
from app.modules.config.constants import UserRole, SensitivityLevel, LLMProvider

role = UserRole.SUPER_ADMIN.value  # "SuperAdmin"
sens = SensitivityLevel.PUBLIC_INTERNAL.value  # "public_internal"
```

---

## Logging

### Logger Setup (per module)
```python
import logging
logger = logging.getLogger(__name__)
```

### Log Levels
- `logger.info(...)` — normal operations, startup events
- `logger.warning(...)` — recoverable issues (fallback used, parse failed)
- `logger.exception("Message: %s", e)` — unexpected errors (includes traceback)
- `logger.error(...)` — errors without traceback needed

### Structured Logging Pattern
Use `%s` formatting (not f-strings) in logger calls for lazy evaluation:
```python
logger.info("RAG Query: conversation_id=%s, provider=%s", req.conversation_id, model_provider)
logger.exception("RAG query failed: %s", e)
```

---

## Testing Patterns

### Container Override for Tests
```python
from app.modules.integration import get_container, reset_container

def setup():
    reset_container()
    container = get_container()
    container.override_instance("vector_store", MockVectorStore())
    container.initialize()
```

### Test Organization
- Integration tests in `test_module/` with `conftest.py` fixtures
- One test file per module: `test_rbac_comprehensive.py`, `test_session_manager.py`, etc.
- Use `pytest-asyncio` for async endpoint tests
- Script-level smoke tests in `scripts/test_*.py`

---

## Document & Data Patterns

### Document Metadata Defaults
Always set defaults before persisting:
```python
metadata.setdefault("department", DEFAULT_DEPARTMENT)
metadata.setdefault("sensitivity", DEFAULT_SENSITIVITY)
metadata["ingested_by"] = requester.get("user_id")
```

### Non-Destructive Versioning
Never modify existing documents. Always create a new version:
```python
result = await document_manager.update_document_version(
    document_id=doc_id,
    text=new_text,
    metadata=updated_metadata,
    version_notes=notes,
    requester_id=user_id,
    status=status
)
```

### File Upload Handling
```python
raw = await file.read()
if not raw:
    raise HTTPException(status_code=400, detail=HTTP_MESSAGES["FILE_EMPTY"])
if len(raw) > MAX_FILE_SIZE_BYTES:
    raise HTTPException(status_code=413, detail=HTTP_MESSAGES["FILE_TOO_LARGE"])
text = raw.decode("utf-8", errors="ignore")
```
