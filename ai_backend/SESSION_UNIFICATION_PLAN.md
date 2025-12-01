# Session Management Unification - Implementation Plan

**Status**: ✅ Already Unified Across All Providers  
**Action Required**: Clean up legacy LangChain code  
**Date**: 2025-12-01

---

## Current State: Already Unified! 🎉

### All RAG Providers Use the Same Session System

```
┌─────────────────────────────────────────────────────────────┐
│                    BaseRAGService                           │
│  (Unified session management for all providers)             │
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │ inject_personalized_context()                     │    │
│  │  ├─ fetch_recent_messages(session_id, limit=2)    │    │
│  │  ├─ Extract tone from history                     │    │
│  │  ├─ get_full_profile(session_id)                  │    │
│  │  └─ Build optimized prefix (80 tokens max)        │    │
│  └───────────────────────────────────────────────────┘    │
│                                                             │
│  ┌───────────────────────────────────────────────────┐    │
│  │ filter_documents_by_rbac()                        │    │
│  │  └─ Version deduplication + RBAC filtering        │    │
│  └───────────────────────────────────────────────────┘    │
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

### Evidence: All Providers Call `query_rag()`

**LocalRAGService** (`rag_local_service.py:680-702`)
```python
async def query_local_rag(..., session_id: Optional[str] = None):
    return await _local_rag_service.query_rag(
        query_text=query_text,
        ...
        session_id=session_id  # ← Passed to BaseRAGService
    )
```

**GoogleRAGService** (`google_models.py:230-249`)
```python
async def query_google_rag(..., session_id: Optional[str] = None):
    return await _google_rag_service.query_rag(
        query_text=query_text,
        ...
        session_id=session_id  # ← Passed to BaseRAGService
    )
```

**GPTRAGService** (`gpt_rag_service.py:125-145`)
```python
async def query_gpt_rag(..., session_id: Optional[str] = None):
    return await _gpt_rag_service.query_rag(
        query_text=query_text,
        ...
        session_id=session_id  # ← Passed to BaseRAGService
    )
```

**HuggingFaceRAGService** (`hf_rag_service.py:86-106`)
```python
async def query_hf_rag(..., session_id: Optional[str] = None):
    return await _hf_rag_service.query_rag(
        query_text=query_text,
        ...
        session_id=session_id  # ← Passed to BaseRAGService
    )
```

---

## Shared Session Management Flow

### 1. Session Creation (support_chat.py)
```python
# When user starts conversation
session_id = create_session(None, role="Employee", department="Engineering")
# → Creates entry in support_sessions.db
```

### 2. Message Storage (support_chat.py)
```python
# User sends message
store_message(session_id, "user", "What is our policy?")
# → Stores in messages table
# → Computes sentiment & tone automatically

# Assistant responds
store_message(session_id, "assistant", "Our policy states...")
# → Stores in messages table
```

### 3. RAG Query (Any Provider)
```python
# User queries any provider (local/google/gpt/hf)
result = await query_[provider]_rag(
    query_text="What is our policy?",
    session_id=session_id,  # ← Session ID passed through
    requester={"user_id": "u123", "role": "Employee", ...}
)

# Inside BaseRAGService.query_rag():
# 1. Retrieve documents
# 2. Filter by RBAC
# 3. inject_personalized_context() ← Fetches session history
# 4. generate_response() ← Provider-specific LLM call
```

### 4. Context Injection (base_rag_service.py)
```python
def inject_personalized_context(session_id, ...):
    # Fetch last 2 messages from SQLite
    chat_history = fetch_recent_messages(session_id, limit=2)
    
    # Extract tone
    for m in reversed(chat_history):
        if m.get("speaker") == "user" and m.get("tone"):
            last_user_tone = m["tone"]
            break
    
    # Get profile if available
    user_profile = get_full_profile(session_id)
    
    # Build optimized prefix (60-80 tokens)
    # Format: "Assistant for Saarthi | Employee/Engineering | John Doe | ..."
```

---

## What ALL Providers Share

### ✅ Session History
- **Source**: `support_chat.fetch_recent_messages(session_id, limit=2)`
- **Storage**: SQLite (`support_sessions.db`)
- **Persistence**: Survives server restarts
- **Isolation**: Per-session (multi-user safe)

### ✅ Sentiment & Tone Tracking
- **Computed**: Automatically on `store_message()`
- **Used For**: Response adaptation via `build_tone_guidance()`
- **Available To**: All providers

### ✅ User Profile Integration
- **Source**: `get_full_profile(session_id)` or `get_all_user_meta(user_id)`
- **Contains**: Name, position, preferences, etc.
- **Used In**: Personalized prompt prefixes

### ✅ RBAC Filtering
- **Method**: `filter_documents_by_rbac()`
- **Rules**: Role hierarchy + department restrictions + allowed_roles overrides
- **Applies To**: All retrieved documents before LLM generation

### ✅ Token Budgeting
- **Budget**: 80 tokens for system prefix
- **Strategy**: Prioritize essential info (role > profile > tone > history)
- **Result**: 60-80 token prefixes vs 200+ previously

### ✅ Audit Logging
- **Events**: `log_user_action()`, `log_security_event()`, `log_performance_metric()`
- **Captured**: User ID, session ID, role, department, query details
- **Provider**: Included in all logs

---

## Legacy Code to Remove

### 🔴 google_models.py (Lines 119-150)

**Current Code** (NOT used in RAG flow):
```python
# --- NEW: LangChain Conversational Chain with Memory ---

try:
    # We create a single, shared memory object for this simple example.
    # In a real multi-user app, you'd manage one memory object per user session.
    chat_memory = ConversationBufferMemory()  # ← REMOVE

    # The ConversationChain is simpler than LLMChain; it has a default prompt.
    conversation_chain = ConversationChain(  # ← REMOVE
        llm=google_llm,
        memory=chat_memory,
        verbose=True
    )

except Exception as e:
    logger.warning(f"Could not initialize Google conversation chain. Error: {e}")
    conversation_chain = None


def get_chat_response(request: ChatRequest) -> ChatResponse:  # ← REMOVE
    """Generates a conversational response using a chain with memory."""
    if not conversation_chain:
        raise ConnectionError("Google Conversation Chain is not initialized.")

    try:
        ai_message = conversation_chain.predict(input=request.user_input)
        return ChatResponse(ai_response=ai_message)

    except Exception as e:
        raise ConnectionError(f"Failed to get response from Google conversation chain: {e}")
```

**Problems**:
1. ❌ Single shared memory (not session-aware)
2. ❌ In-memory only (lost on restart)
3. ❌ Not integrated with RAG endpoints
4. ❌ Bypasses RBAC filtering
5. ❌ No audit logging
6. ❌ No token budgeting

**Used By**: Nothing (orphaned code)

---

## Cleanup Actions

### Step 1: Remove Unused Imports from google_models.py

**Remove** (lines 6-7):
```python
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
```

**Keep**:
```python
from langchain.prompts import PromptTemplate  # Still used for idea generation
from langchain_google_genai import ChatGoogleGenerativeAI  # Core LLM client
```

### Step 2: Remove Unused Functions from google_models.py

**Remove** (lines 119-150):
- `chat_memory = ConversationBufferMemory()`
- `conversation_chain = ConversationChain(...)`
- `get_chat_response(request: ChatRequest)` function

### Step 3: Remove Unused Models from google_models.py

**Remove** (lines 20-25):
```python
class ChatRequest(BaseModel):
    user_input: str = Field(..., min_length=1, description="The user's message to the chatbot.")

class ChatResponse(BaseModel):
    ai_response: str
```

**Reason**: Only used by removed `get_chat_response()` function

### Step 4: Update API Routes (if any)

**Check**: `api_routes_*.py` files for any endpoints calling `get_chat_response()`

**Expected**: None (this was example code)

### Step 5: Update requirements.txt

**Optional**: If not using LangChain chains anywhere else, update dependencies

**Check**:
```bash
# See if LangChain chains are used elsewhere
grep -r "LLMChain\|ConversationChain" app/services/
```

**Keep**: `langchain` for `PromptTemplate` and `ChatGoogleGenerativeAI`

---

## Verification Steps

### 1. Test All Providers with Same Session

```python
# Test script: test_unified_sessions.py

import asyncio
from app.services.support_chat import create_session, store_message, fetch_recent_messages
from app.services.rag_local_service import query_local_rag
from app.services.google_models import query_google_rag
from app.services.gpt_rag_service import query_gpt_rag
from app.services.hf_rag_service import query_hf_rag

async def test_session_unification():
    # Create session
    session_id = create_session(None, "Employee", "Engineering")
    print(f"Created session: {session_id}")
    
    # Simulate conversation
    store_message(session_id, "user", "Hello, what is our company policy?")
    store_message(session_id, "assistant", "Let me check that for you.")
    
    requester = {
        "user_id": "u_test",
        "role": "Employee",
        "department": "Engineering"
    }
    
    # Test each provider uses same session
    providers = [
        ("Local", query_local_rag),
        ("Google", query_google_rag),
        ("GPT", query_gpt_rag),
        ("HuggingFace", query_hf_rag)
    ]
    
    for name, query_func in providers:
        try:
            result = await query_func(
                query_text="What is our vacation policy?",
                session_id=session_id,  # ← Same session for all
                requester=requester,
                use_llm=True
            )
            print(f"\n{name} Provider:")
            print(f"  - Answer: {result['answer'][:100]}...")
            print(f"  - Used session: {session_id}")
            
            # Verify session history was used
            history = fetch_recent_messages(session_id)
            print(f"  - History length: {len(history)}")
            
        except Exception as e:
            print(f"{name} Provider: SKIPPED ({e})")
    
    # Verify all providers added to same session
    final_history = fetch_recent_messages(session_id, limit=20)
    print(f"\nFinal session history: {len(final_history)} messages")
    
    return session_id

# Run test
if __name__ == "__main__":
    asyncio.run(test_session_unification())
```

### 2. Verify Session Persistence

```python
# Test that sessions persist across "restarts"

async def test_session_persistence():
    # Create session and add messages
    session_id = create_session(None, "Manager", "HR")
    store_message(session_id, "user", "First message")
    
    # Query with local provider
    await query_local_rag(
        query_text="Test query",
        session_id=session_id,
        requester={"user_id": "u1", "role": "Manager", "department": "HR"}
    )
    
    # Simulate restart (reconnect to DB)
    # Session should still exist
    history = fetch_recent_messages(session_id)
    assert len(history) >= 1, "Session history lost!"
    
    # Query with different provider
    await query_google_rag(
        query_text="Another test",
        session_id=session_id,
        requester={"user_id": "u1", "role": "Manager", "department": "HR"}
    )
    
    # Verify both queries in same session
    final_history = fetch_recent_messages(session_id)
    assert len(final_history) > len(history), "Second query not in same session!"
    
    print("✅ Session persistence verified across providers")
```

### 3. Verify No LangChain Memory Usage

```bash
# Ensure no ConversationBufferMemory in active code paths
grep -r "ConversationBufferMemory" app/services/*.py | grep -v "^#"

# Expected output: Empty (or only in comments)
```

---

## Benefits of Current Unified System

### ✅ Single Source of Truth
- All conversation history in one SQLite database
- Consistent across all providers
- No synchronization issues

### ✅ Provider Agnostic
- Switch providers mid-conversation
- Same session ID works everywhere
- Consistent user experience

### ✅ Production Ready
- Persistent storage
- Multi-user isolation
- Comprehensive audit logs
- RBAC integration

### ✅ Optimized Performance
- Token budgeting (80 tokens max)
- Efficient history fetching (last 2 messages)
- Smart truncation strategies

### ✅ Enterprise Features
- Sentiment tracking
- Profile personalization
- Tone adaptation
- Security event logging

---

## Migration Checklist

- [ ] **Step 1**: Review current architecture (✅ Already unified!)
- [ ] **Step 2**: Remove LangChain `ConversationBufferMemory` from `google_models.py`
- [ ] **Step 3**: Remove unused imports and functions
- [ ] **Step 4**: Run integration tests (all providers with same session)
- [ ] **Step 5**: Verify no LangChain memory in codebase
- [ ] **Step 6**: Update documentation in `APP_CONTEXT.md`
- [ ] **Step 7**: Deploy and monitor

---

## Documentation Updates Needed

### Update APP_CONTEXT.md

Add section:

```markdown
## Unified Session Management

All RAG providers (Local, Google, GPT, HuggingFace) use the same session management system:

**Architecture**:
- Centralized in `BaseRAGService`
- SQLite-backed persistence (`support_sessions.db`)
- Session-isolated conversation history
- Automatic sentiment & tone tracking
- Token-budgeted prompt optimization

**Session Flow**:
1. Create session: `create_session(None, role, department)`
2. Store messages: `store_message(session_id, speaker, content)`
3. Query any provider: Pass `session_id` parameter
4. BaseRAGService automatically:
   - Fetches recent messages (2 most recent)
   - Extracts user tone
   - Loads user profile
   - Builds optimized prefix (60-80 tokens)
   - Applies RBAC filtering
   - Logs all actions

**Benefits**:
- Switch providers mid-conversation seamlessly
- Persistent history across server restarts
- Multi-user isolation
- Comprehensive audit trails
```

---

## Conclusion

✅ **Your system is already unified!**

All RAG providers use the same SQLite-based session management through `BaseRAGService`. The only cleanup needed is removing the legacy LangChain `ConversationBufferMemory` code in `google_models.py` that's not integrated with your main RAG flow.

**No migration required** - just cleanup to remove confusing legacy code.

**Next Steps**:
1. Remove legacy LangChain code from `google_models.py`
2. Run integration tests to verify unified behavior
3. Update documentation to highlight unified architecture

---

**Document Version**: 1.0  
**Status**: Implementation Ready  
**Last Updated**: 2025-12-01
