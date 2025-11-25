# API Testing Guide

Here are **ready-to-run cURL examples** for testing the AI Engineering API.

## Table of Contents
1. [Support Chat & Onboarding](#1-support-chat--onboarding)
2. [RAG Services](#2-rag-services)
3. [LLM Services](#3-llm-services)
4. [Sentiment Analysis](#4-sentiment-analysis)

---

## 1. Support Chat & Onboarding

### Start a Session
```bash
curl -X POST http://localhost:8000/api/rag/session/start
```
*Returns `session_id`. Use this in headers for subsequent requests.*

### End a Session
```bash
curl -X POST http://localhost:8000/api/rag/session/end \
  -H "Content-Type: application/json" \
  -d '{"session_id": "sess_xxxxx"}'
```

---

## 2. RAG Services

### Query (Local Model)
*Supports RBAC and session context.*

```bash
curl -X POST http://localhost:8000/api/rag/local/query \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: sess_xxxxx" \
  -H "X-API-Key: employee_key" \
  -d '{
        "question": "What are the company leave policies?",
        "top_k": 3,
        "use_llm": true
      }'
```

### Query (Google Model)
```bash
curl -X POST http://localhost:8000/api/rag/google/query \
  -H "Content-Type: application/json" \
  -H "X-Session-Id: sess_xxxxx" \
  -d '{
        "question": "Explain the project roadmap.",
        "use_llm": true
      }'
```

### Add Document (JSON)
```bash
curl -X POST http://localhost:8000/api/rag/add \
  -H "Content-Type: application/json" \
  -H "X-API-Key: manager_key" \
  -d '{
        "source_name": "policy_update.txt",
        "text": "New policy: Fridays are half-days.",
        "metadata": {"department": "HR", "sensitivity": "public_internal"}
      }'
```

### Add Document (File Upload)
*Supports .md, .html, .json, .txt. Automatically parses content.*

```bash
curl -X POST http://localhost:8000/api/rag/add-file \
  -H "X-API-Key: manager_key" \
  -F "file=@/path/to/document.md" \
  -F "department=Engineering"
```

### Seed Default Data
```bash
curl -X POST http://localhost:8000/api/rag/seed?reseed=true
```

### Clear Collection (Exec/Legal only)
```bash
curl -X POST http://localhost:8000/api/rag/clear \
  -H "X-API-Key: executive_key"
```

---

## 3. LLM Services

### Chat (Conversational)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Hello, who are you?"}'
```

### Generate Content Ideas
```bash
curl -X POST http://localhost:8000/generate/ideas \
  -H "Content-Type: application/json" \
  -d '{"topic": "Future of AI in Healthcare"}'
```

### Simple Text Generation
```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Write a haiku about coding."}'
```

### Summarization
```bash
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"text": "Long text content here..."}'
```

---

## 4. Sentiment Analysis

### Analyze Text
```bash
curl -X POST http://localhost:8000/api/rag/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "I am frustrated with this error!"}'
```

### Get Stats
```bash
curl http://localhost:8000/api/rag/sentiment/stats
```

---

## RAG Conversation Flow Diagram

```mermaid
flowchart TD
  Start([Start]) --> StartSession["POST /api/rag/session/start\n(x-api-key)"]
  StartSession -->|200: {session_id, message}| SessionCreated["Session Created\nstore session_id"]

  subgraph User Conversation Loop
    direction TB
    UserQuery["POST /api/rag/local/query\n(x-session-id, x-api-key)\n{question, top_k, use_llm, max_tokens, category}"]
    RAGResponse["RAG Response\n{answer, retrieved, context}"]
    UserQuery --> RAGResponse
    RAGResponse --> CheckIfFinal{Is answer ==\n'Thank you! Your details have been saved.'?}
    CheckIfFinal -->|No| UserProvidesAnswer["User replies (next query)\nrepeat loop"]
    CheckIfFinal -->|Yes| Finalized["Persist collected details\nreturn final ack"]
    UserProvidesAnswer --> UserQuery
  end

  SessionCreated --> UserQuery
  Finalized --> LLMFollowup["POST /api/rag/local/query\n(use_llm: true)\n{AI-generated followup question}"]
  LLMFollowup --> End([End / Next: Chat App Integration])
```