# API Testing Guide

Here are **ready-to-run cURL examples** for testing the AI Engineering API.

## Table of Contents
1. [Authentication](#1-authentication)
2. [RAG Services](#2-rag-services)
3. [LLM Services](#3-llm-services)
4. [Sentiment Analysis](#4-sentiment-analysis)
  "token_type": "bearer",
  "user": {
    "user_id": "u_admin_1",
    "username": "admin",
    "role": "SuperAdmin",
    "department": "Executive",
    "profile": {
      "name": "Admin User",
      "email": "admin@company.com"
    }
  }
}
```

**Set token as environment variable:**
```bash
export TOKEN="your_jwt_token_here"
```

**Note:** Sessions are now managed automatically via JWT tokens. The `user_id` from the token is used as the session identifier. No separate session start/end endpoints needed.

---

## 2. RAG Services

### Query (Local Model - Authenticated)
*Supports RBAC and session context via JWT token.*

```bash
curl -X POST http://localhost:5444/api/rag/local/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
        "question": "What are the company leave policies?",
        "top_k": 3,
        "use_llm": true
      }'
```

### Query (Guest/Unauthenticated)
*Public endpoint - no authentication required. Returns only public documents.*

```bash
curl -X POST http://localhost:5444/api/rag/local/query \
  -H "Content-Type: application/json" \
  -d '{
        "question": "What are your office hours?",
        "top_k": 3,
        "use_llm": true
      }'
```

### Query (Google Model)
```bash
curl -X POST http://localhost:5444/api/rag/google/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
### Chat (Conversational)
```bash
curl -X POST http://localhost:5444/chat \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Hello, who are you?"}'
```

### Generate Content Ideas
```bash
curl -X POST http://localhost:5444/generate/ideas \
  -H "Content-Type: application/json" \
  -d '{"topic": "Future of AI in Healthcare"}'
```

### Simple Text Generation
```bash
curl -X POST http://localhost:5444/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Write a haiku about coding."}'
```

### Summarization
```bash
curl -X POST http://localhost:5444/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Long text content here..."}'
```

---

## 4. Sentiment Analysis

*SuperAdmin only*

### Analyze Text
```bash
curl -X POST http://localhost:5444/api/rag/sentiment \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"text": "I am frustrated with this error!"}'
```

### Get Stats
```bash
curl -X GET http://localhost:5444/api/rag/sentiment/stats \
  -H "Authorization: Bearer $TOKEN"
  -H "Authorization: Bearer $TOKEN" \
  -d '{
        "document_id": "doc_abc123...",
        "text": "Company Leave Policy v2.0\n\nEmployees get 20 days annual leave (updated!).",
        "version_notes": "Increased annual leave from 15 to 20 days",
        "status": "published"
      }'
```

### List All Documents
```bash
curl -X GET "http://localhost:5444/api/rag/documents/list?latest_only=true" \
  -H "Authorization: Bearer $TOKEN"
```

### List by Department
```bash
curl -X GET "http://localhost:5444/api/rag/documents/list?department=HR&latest_only=true" \
  -H "Authorization: Bearer $TOKEN"
```

### Get Version History
```bash
curl -X GET "http://localhost:5444/api/rag/documents/doc_abc123.../versions" \
  -H "Authorization: Bearer $TOKEN"
```

### Get Specific Version
```bash
curl -X GET "http://localhost:5444/api/rag/documents/doc_abc123.../versions/1.0" \
  -H "Authorization: Bearer $TOKEN"
```

### Compare Two Versions
```bash
curl -X GET "http://localhost:5444/api/rag/documents/doc_abc123.../compare?version1=1.0&version2=2.0" \
  -H "Authorization: Bearer $TOKEN"
```

### Archive a Version
```bash
curl -X POST "http://localhost:5444/api/rag/documents/doc_abc123.../archive" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"version": "1.0"}'
```

### Query (With Versioning - Returns Latest Only)
```bash
curl -X POST http://localhost:5444/api/rag/local/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
        "question": "What is the leave policy?",
        "top_k": 3,
        "use_llm": true
      }'
```
*Note: Query automatically filters for latest published versions only*