# Architecture Improvements

## 1. Chain of Responsibility for Prompt Building

**Before:**
```python
# Rigid, hard to extend
system_prompt = await self.prompt_manager.build_system_prompt(request.user, context, request.category)
user_prompt = await self.prompt_manager.build_user_prompt(request.question, context)
final_prompt = await self.prompt_manager.build_full_prompt(system_prompt, user_prompt)
```

**After:**
```python
# Flexible, extensible chain
final_prompt = await self.prompt_chain.build_prompt(
    user=request.user,
    question=request.question,
    context=context,
    category=request.category
)
```

**Adding New Prompt Layer:**
```python
class ComplianceHandler(PromptHandler):
    """Adds compliance instructions."""
    
    async def process(self, context: PromptContext) -> PromptContext:
        if context.category == "legal":
            context.system_prompt += " Ensure all advice complies with current regulations."
        return context

# Add to chain
chain.add_handler(ComplianceHandler(), position=2)
```

## 2. Plugin-Based Provider Factory

**Before:**
```python
# Hard-coded provider creation
provider = ProviderFactory.create_provider(request.provider, model_name)
```

**After:**
```python
# Plugin-based, extensible
provider = await create_provider(request.provider, config)
```

**Adding New Provider:**
```python
class AnthropicProviderPlugin(ProviderPlugin):
    @property
    def name(self) -> str:
        return "anthropic"
    
    async def create_provider(self, config=None):
        return AnthropicProvider()

# Register plugin
registry = get_provider_registry()
registry.register(AnthropicProviderPlugin())
```

## 3. Middleware Pattern for Cross-Cutting Concerns

**Before:**
```python
# Cross-cutting concerns mixed with business logic
def process_query(self, request):
    start_time = time.time()  # Timing
    logger.info("Processing request")  # Logging
    # ... business logic
    duration = time.time() - start_time
    logger.info(f"Took {duration}s")
```

**After:**
```python
# Clean separation of concerns
request = await self.middleware_stack.process_request(request)
response = await self._core_processing(request)
return await self.middleware_stack.process_response(request, response)
```

**Adding New Middleware:**
```python
class AuditMiddleware(RAGMiddleware):
    async def process_request(self, request: RAGRequest) -> RAGRequest:
        audit_log.record_request(request.user, request.question)
        return request
    
    async def process_response(self, request: RAGRequest, response: RAGResponse) -> RAGResponse:
        audit_log.record_response(request.user, response.answer)
        return response

# Add to stack
orchestrator.middleware_stack.add(AuditMiddleware())
```

## 4. Benefits of New Architecture

### **Extensibility**
- Add new prompt handlers without changing existing code
- Register new providers as plugins
- Insert middleware for new features

### **Maintainability**
- Single responsibility principle
- Clear separation of concerns
- Easy to test individual components

### **Flexibility**
- Configure chains at runtime
- Enable/disable middleware dynamically
- Swap providers without code changes

### **Testability**
- Mock individual handlers/middleware
- Test chains in isolation
- Unit test each component

## 5. Usage Examples

### Custom Prompt Chain
```python
# Create specialized chain for legal queries
legal_chain = PromptChain()
legal_chain.add_handler(SystemPromptHandler())
legal_chain.add_handler(ComplianceHandler())
legal_chain.add_handler(LegalDisclaimerHandler())
legal_chain.add_handler(FinalPromptHandler())

# Use for legal category
if request.category == "legal":
    prompt = await legal_chain.build_prompt(...)
```

### Conditional Middleware
```python
# Add performance monitoring only in production
if settings.ENVIRONMENT == "production":
    orchestrator.middleware_stack.add(PerformanceMiddleware())
    orchestrator.middleware_stack.add(MetricsMiddleware())
```

### Provider Fallback
```python
# Try primary provider, fallback to secondary
try:
    provider = await create_provider("openai", config)
except Exception:
    provider = await create_provider("local", config)
```

## 6. Future Improvements

1. **Event-Driven Architecture**: Decouple components with events
2. **Circuit Breaker Pattern**: Handle provider failures gracefully
3. **Strategy Pattern**: Different retrieval strategies
4. **Observer Pattern**: React to system events
5. **Command Pattern**: Queue and retry operations

## 7. Migration Path

1. **Phase 1**: Implement new patterns alongside existing code
2. **Phase 2**: Gradually migrate endpoints to use new architecture
3. **Phase 3**: Remove old implementations
4. **Phase 4**: Add advanced patterns (events, circuit breakers)

The new architecture maintains backward compatibility while providing a clear path for future enhancements.