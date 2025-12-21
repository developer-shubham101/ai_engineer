# Development Guidelines & Standards

## Code Quality Standards

### Import Organization
- **Future Imports First**: Use `from __future__ import annotations` for forward compatibility
- **Standard Library**: Import standard library modules before third-party
- **Third-Party Libraries**: Group FastAPI, Pydantic, and other external dependencies
- **Local Imports**: Import application modules last with relative imports
- **Conditional Imports**: Place imports inside functions when needed for optional dependencies

### Type Annotations
- **Comprehensive Typing**: All function parameters and return types must be annotated
- **Generic Types**: Use `Dict[str, Any]`, `List[Dict[str, Any]]`, `Optional[str]` consistently
- **Interface Types**: Prefer interface types over concrete implementations in signatures
- **Future Annotations**: Enable `from __future__ import annotations` for forward references

### Error Handling Patterns
- **HTTPException Usage**: Raise `HTTPException` with appropriate status codes and detail messages
- **Exception Chaining**: Use `raise HTTPException` after logging exceptions with `logger.exception()`
- **Graceful Degradation**: Handle missing services/models with fallback behavior
- **Validation Errors**: Convert validation errors to HTTP 400 with descriptive messages

## Architectural Patterns

### Dependency Injection
- **Container Pattern**: Use `get_container()` to access modular services
- **Interface Segregation**: Depend on interfaces (`ILLMProvider`, `IRBACManager`) not implementations
- **Service Initialization**: Call `container.initialize()` before accessing services
- **Lazy Loading**: Initialize expensive resources only when needed

### Modular Service Architecture
```python
# Standard pattern for accessing services
container = get_container()
container.initialize()
service: IServiceInterface = container.get_service()
```

### Provider Factory Pattern
- **Dynamic Provider Selection**: Use factory pattern for LLM providers based on runtime parameters
- **Extensible Design**: New providers implement common interface without changing existing code
- **Configuration-Driven**: Provider selection based on configuration or request parameters
- **Fallback Mechanisms**: Graceful handling when preferred providers unavailable

### Authentication & Authorization
- **Dependency Injection**: Use `Depends(get_current_user)` for authenticated endpoints
- **Role-Based Access**: Apply `Depends(require_roles(ROLE_LIST))` for authorization
- **Optional Authentication**: Use `get_current_user_optional` for endpoints supporting both modes
- **RBAC Integration**: Check permissions through `IRBACManager` interface

## API Design Standards

### FastAPI Route Structure
- **Router Organization**: Group related endpoints in separate router files (`api_routes_*.py`)
- **Prefix Consistency**: Use constants for API prefixes (`RAG_PREFIX`, `API_PREFIX`)
- **Tag Organization**: Apply consistent tags for OpenAPI documentation grouping
- **Response Models**: Define Pydantic models for all response types

### Request/Response Models
- **Pydantic Models**: All request/response bodies use Pydantic models with validation
- **Default Values**: Provide sensible defaults using constants (`DEFAULT_TOP_K`, `DEFAULT_MAX_TOKENS`)
- **Optional Fields**: Use `Optional[Type]` for non-required fields
- **Field Validation**: Use `Field()` for additional validation and documentation

### Error Response Patterns
```python
# Standard error handling pattern
try:
    result = await service.operation()
    return SuccessResponse(data=result)
except ValueError as e:
    raise HTTPException(status_code=404, detail=str(e))
except Exception as e:
    logger.exception("Operation failed: %s", e)
    raise HTTPException(status_code=500, detail=str(e))
```

## Security Implementation

### Role-Based Access Control (RBAC)
- **Hierarchical Roles**: Use numeric levels (0-4) for role hierarchy comparison
- **Document-Level Security**: Filter documents based on sensitivity and user permissions
- **Department Restrictions**: Enforce department-based access for confidential documents
- **Audit Logging**: Log all access attempts for security compliance

### Metadata Validation
- **Sensitivity Levels**: Validate against `VALID_SENSITIVITY_LEVELS` constants
- **Department Validation**: Check against `VALID_DEPARTMENTS` list
- **User Permission Checks**: Verify user can create documents with specified sensitivity
- **Override Mechanisms**: Support `allowed_roles` for flexible access control

### Security Logging
```python
# Standard security logging patterns
from app.logging_config import log_security_event, log_user_action

log_security_event(logger, "ACCESS_DENIED", user_id, 
                  role=user_role, resource=resource)
log_user_action(logger, "DOCUMENT_CREATED", user_id,
               document_id=doc_id, sensitivity=sensitivity)
```

## Data Management Patterns

### Document Versioning
- **Non-Destructive Updates**: Create new versions instead of modifying existing documents
- **Version History**: Maintain complete audit trail of document changes
- **Metadata Preservation**: Carry forward metadata with version-specific overrides
- **Status Management**: Track document lifecycle with status fields

### Database Interactions
- **Service Layer**: Access databases through service interfaces, not direct connections
- **Transaction Management**: Use appropriate transaction boundaries for multi-step operations
- **Error Recovery**: Handle database errors gracefully with meaningful user messages
- **Connection Pooling**: Rely on service layer for connection management

## Performance Optimization

### Token Management
- **Token Estimation**: Use `estimate_tokens_from_text()` for prompt planning
- **Context Window Limits**: Check against model context limits before generation
- **Smart Truncation**: Implement intelligent context truncation for long documents
- **Budget Allocation**: Balance tokens between system prompts, context, and generation

### Caching Strategies
- **Model Caching**: Cache loaded models to avoid repeated initialization
- **Embedding Caching**: Store computed embeddings for reuse
- **Instance Reuse**: Maintain service instances across requests where appropriate
- **Lazy Loading**: Initialize expensive resources only when needed

### Logging & Monitoring
```python
# Performance logging pattern
from app.logging_config import log_performance_metric, log_llm_interaction

start_time = time.time()
result = await operation()
duration = (time.time() - start_time) * 1000

log_performance_metric(logger, "OPERATION_NAME", duration,
                      additional_metrics=metrics)
```

## Testing Standards

### Test Organization
- **Module-Based Testing**: Organize tests by module in `tests/` directory
- **Integration Tests**: Separate integration tests in `test_module/` directory
- **Fixture Usage**: Use pytest fixtures for common test setup
- **Async Testing**: Use `pytest-asyncio` for testing async functions

### Mock Patterns
- **Service Mocking**: Mock service interfaces rather than implementations
- **External API Mocking**: Mock external API calls for reliable testing
- **Database Mocking**: Use in-memory databases or mocks for unit tests
- **Configuration Mocking**: Override configuration for test scenarios

## Configuration Management

### Environment Variables
- **Typed Configuration**: Use Pydantic models for configuration validation
- **Default Values**: Provide sensible defaults for all configuration options
- **Environment Separation**: Support different configurations for dev/test/prod
- **Secret Management**: Handle API keys and secrets securely

### Constants Organization
- **Centralized Constants**: Define constants in `app/modules/config/constants.py`
- **Grouped Constants**: Organize related constants together (roles, sensitivity levels)
- **Type Safety**: Use enums or typed constants where appropriate
- **Documentation**: Document the purpose and valid values for constants

## Documentation Standards

### Code Documentation
- **Docstring Format**: Use triple-quoted strings with clear descriptions
- **Parameter Documentation**: Document all parameters and return values
- **Example Usage**: Include usage examples for complex functions
- **Type Information**: Complement type hints with docstring descriptions

### API Documentation
- **OpenAPI Integration**: Leverage FastAPI's automatic OpenAPI generation
- **Response Examples**: Provide example responses in route documentation
- **Error Documentation**: Document possible error responses and status codes
- **Tag Organization**: Use consistent tags for logical API grouping