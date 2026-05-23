# Query Chain Pattern Example

## Usage

**Before:**
```python
# Simple query
query = "What is our vacation policy?"
documents = await retrieve_documents(query, user, top_k=3)
```

**After:**
```python
# Enhanced query with chain
enhanced_query = await query_chain.build_query(
    query="What is our vacation policy?",
    user={"role": "Employee", "department": "Engineering"},
    session_id="sess_123",
    category="hr"
)
# Result: "What is our vacation policy? [User: Employee in Engineering] [Mood: frustrated] [Recent: asked about sick leave] [Category: hr]"
```

## Adding Custom Handlers

```python
class CompanyContextHandler(QueryHandler):
    """Adds company-specific context."""
    
    async def process(self, context: QueryContext) -> QueryContext:
        if "policy" in context.original_query.lower():
            context.enhanced_query += " [Context: Company policies updated Q4 2024]"
        return context

# Add to chain
query_chain.add_handler(CompanyContextHandler())
```

## Conditional Handlers

```python
class UrgencyHandler(QueryHandler):
    """Adds urgency markers."""
    
    async def process(self, context: QueryContext) -> QueryContext:
        urgent_words = ["urgent", "asap", "emergency"]
        if any(word in context.original_query.lower() for word in urgent_words):
            context.enhanced_query += " [URGENT]"
        return context

# Only add for certain roles
if user.get("role") in ["Manager", "SuperAdmin"]:
    query_chain.add_handler(UrgencyHandler())
```

## Benefits

1. **Modular**: Add/remove query enhancement components
2. **Contextual**: Automatically includes relevant context
3. **Extensible**: Easy to add new enhancement logic
4. **Configurable**: Enable/disable handlers per user/session