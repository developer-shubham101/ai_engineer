# 🔌 API Documentation & Testing Guide

**Base URL:** `http://127.0.0.1:8000`

## Table of Contents

1. [General Endpoints](#general-endpoints)
2. [Authentication](#authentication)
3. [RAG Query APIs](#rag-query-apis)
4. [Document Management](#document-management)
5. [Agents API](#agents-api)
6. [Conversation History](#conversation-history)
7. [Prompt Templates](#prompt-templates)
8. [Multimodal](#multimodal)
9. [Model Management](#model-management)
10. [Document Cleanup & Metadata Enrichment](#document-cleanup--metadata-enrichment)
11. [System Status](#system-status)
12. [RBAC & Permissions Reference](#rbac--permissions-reference)

---

## General Endpoints

```bash
GET  /              # → {"status": "ok", "message": "Welcome to the AI Engineering API!"}
GET  /health        # → {"status": "healthy", "architecture": "modular"}
GET  /api/modules/status   # check container initialization
GET  /docs          # Swagger UI
```

---

## Authentication

### Login
```bash
POST /api/auth/token
```
```bash
curl -X POST "http://127.0.0.1:8000/api/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'
```
```json
{
  "access_token": "eyJ0eXAiOiJKV1Qi...",
  "token_type": "bearer",
  "user": {
    "user_id": "u_admin_1",
    "username": "admin",
    "role": "SuperAdmin",
    "department": "Executive"
  }
}
```

Use token in all subsequent requests:
```
Authorization: Bearer <access_token>
```

### Roles & Levels
| Role | Level |
|------|-------|
| SuperAdmin | 4 |
| Manager | 3 |
| HR | 2 |
| Employee | 1 |
| Guest | 0 |

---

## RAG Query APIs

### Query Preprocessing (as-you-type)
```bash
POST /api/rag/query/preprocess
```
No auth required. Call debounced on keystroke pause to get a "Did you mean?" suggestion before the user submits.

```bash
curl -X POST "http://127.0.0.1:8000/api/rag/query/preprocess" \
  -H "Content-Type: application/json" \
  -d '{"query": "wht is pto polcy"}'
```
```json
{
  "original":   "wht is pto polcy",
  "corrected":  "what is pto policy",
  "expanded":   "what is pto paid time off vacation leave policy",
  "query_type": "policy",
  "suggestion": "what is pto policy"
}
```
`suggestion` = best single string to show (`rewritten > corrected > expanded`), `null` if nothing changed. The user accepts or ignores it before submitting to `/query`.

---

### RAG Query
```bash
POST /api/rag/{provider}/query
```

**Providers:** `local` · `google` · `gpt` · `openai` · `huggingface` · `hf` · `customllm` · `llamaserver`

Auth is optional — unauthenticated requests are served as Guest.

```bash
curl -X POST "http://127.0.0.1:8000/api/rag/local/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "question": "What is the company vacation policy?",
    "conversation_id": "conv_abc123",
    "top_k": 3,
    "use_documents": true,
    "use_llm": true,
    "use_conversation_history": true,
    "max_tokens": 256,
    "temperature": 0.1,
    "prompt_template": "personalized_chat",
    "local_llm_model": "phi2"
  }'
```

**Request parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | string | required | User question |
| `conversation_id` | string | null | Links query to conversation history |
| `top_k` | int | 3 | Documents to retrieve |
| `use_documents` | bool | true | Run vector+BM25 retrieval |
| `use_llm` | bool | false | Generate LLM answer |
| `use_conversation_history` | bool | true | Include past messages as context |
| `enable_agentic_mode` | bool | false | Add "think step by step" to prompt |
| `use_tools` | bool | false | Run legacy tool-loop agent |
| `max_tokens` | int | 256 | LLM max output tokens |
| `temperature` | float | 0.1 | 0.0 = deterministic, 1.0 = creative |
| `category` | string | null | Metadata category filter |
| `debug` | bool | false | Include `final_prompt` in response |
| `prompt_template` | string | null | DB template name to use |
| `local_llm_model` | string | null | Local provider only — specific GGUF key |

**Response:**
```json
{
  "answer": "The company provides 20 days of annual leave...",
  "retrieved": [
    {
      "id": "doc_123",
      "text": "Leave policy excerpt...",
      "metadata": {"department": "HR", "sensitivity": "public_internal"},
      "distance": 0.85
    }
  ],
  "context": "Document 1: Leave policy...",
  "final_prompt": "SYSTEM: ...\n\nUSER: ..."
}
```

**Retrieval pipeline:** `query → BM25.search() + vector_store.search() → RRF(1.0, 1.0) → RBAC filter → cross-encoder rerank → top-k`

**Provider examples:**
```bash
# Google Gemini
curl -X POST "http://127.0.0.1:8000/api/rag/google/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our mission?", "use_llm": true}'

# OpenAI GPT (aliases: gpt, openai)
curl -X POST "http://127.0.0.1:8000/api/rag/gpt/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain the benefits package", "use_llm": true, "temperature": 0.3}'

# Hugging Face (aliases: huggingface, hf)
curl -X POST "http://127.0.0.1:8000/api/rag/hf/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is PTO?", "use_llm": true}'

# CustomLLM / third-party (preferred for external APIs)
curl -X POST "http://127.0.0.1:8000/api/rag/customllm/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the leave policy?", "use_llm": true}'

# LlamaServer (local llama-server.exe)
curl -X POST "http://127.0.0.1:8000/api/rag/llamaserver/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarize the HR handbook", "use_llm": true}'
```

---

## Document Management

All document endpoints require at minimum `Employee` role.

### Add Document (JSON)
```bash
POST /api/rag/documents/add        # Employee+
```
```bash
curl -X POST "http://127.0.0.1:8000/api/rag/documents/add" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "source_name": "remote_work_policy.md",
    "text": "Employees may work remotely up to 3 days per week...",
    "metadata": {
      "department": "HR",
      "sensitivity": "public_internal",
      "document_type": "policy"
    }
  }'
```

### Upload Document File
```bash
POST /api/rag/documents/add-file   # Employee+
```
Supported: `.txt`, `.md`, `.html`, `.json`. Max 5 MB.
```bash
curl -X POST "http://127.0.0.1:8000/api/rag/documents/add-file" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@policy.md" \
  -F "department=HR" \
  -F "sensitivity=public_internal"
```

### Update Document (creates new version)
```bash
POST /api/rag/documents/update     # Employee+
```
```bash
curl -X POST "http://127.0.0.1:8000/api/rag/documents/update" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "document_id": "doc_abc123",
    "text": "Updated policy content v2...",
    "version_notes": "Increased leave from 15 to 20 days",
    "status": "published"
  }'
```

### Seed, List, Version, Archive
```bash
POST /api/rag/documents/seed                              # SuperAdmin — load companyData/
POST /api/rag/documents/clear                             # SuperAdmin
GET  /api/rag/documents/list?department=HR&latest_only=true   # Employee+
GET  /api/rag/documents/{id}/versions                     # Employee+
GET  /api/rag/documents/{id}/versions/{version}           # Employee+
GET  /api/rag/documents/{id}/compare?version1=v1&version2=v2  # Employee+
POST /api/rag/documents/{id}/archive                      # HR+
```

### Metadata Validation

| Field | Valid values |
|-------|-------------|
| `department` | `General`, `HR`, `Finance`, `Engineering`, `IT`, `Legal`, `Executive`, `Admin` |
| `sensitivity` | `public_internal`, `department_confidential`, `role_confidential`, `highly_confidential`, `super_confidential`, `personal` |
| `allowed_roles` | Array of role strings — bypasses hierarchy entirely |
| `owner_id` | Required for `personal` documents |

---

## Agents API

All agent queries go to the same unified endpoint. `orchestrator_type` selects the engine.

### Query
```bash
POST /api/agents/query
```

```bash
curl -X POST "http://127.0.0.1:8000/api/agents/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "question": "What is Tesla stock price and weather in Austin?",
    "orchestrator_type": "autogen",
    "workflow": "smart_assistant",
    "tools": [],
    "max_steps": 5,
    "temperature": 0.1
  }'
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | string | required | User question or topic |
| `orchestrator_type` | string | `autogen` | `autogen` · `custom` · `mcp` · `crewai` |
| `workflow` | string | `smart_assistant` | See workflow table below |
| `tools` | array | `[]` | Tool names to inject (empty = all) |
| `max_steps` | int | 5 | Max agent steps |
| `temperature` | float | 0.1 | LLM temperature |
| `conversation_id` | string | null | Auto-created if omitted; always returned |
| `debug` | bool | false | Include `debug_info` in response |

**Orchestrators & supported workflows:**

| `orchestrator_type` | Workflows |
|---------------------|-----------|
| `autogen` | `debate`, `research`, `smart_assistant`, `smart_travel_planner`, `prompt_evaluation` |
| `custom` | `debate`, `research`, `smart_assistant`, `smart_travel_planner` |
| `mcp` | `smart_assistant` only |
| `crewai` | `debate`, `research`, `smart_travel_planner` |

**Workflows:**

| Workflow | Agents | Description |
|----------|--------|-------------|
| `smart_assistant` | ToolSelector → ToolExecutor → Summarizer | Auto tool selection + execution |
| `smart_travel_planner` | TravelToolSelector → ToolExecutor → TravelPlanner | Intent-driven travel planning |
| `debate` | Advocate, Critic, Moderator | Multi-perspective debate |
| `research` | Planner, Researcher, Verifier, Analyst, Evaluator, ReportWriter | 6-agent research pipeline |
| `prompt_evaluation` | PromptParser, CriteriaJudge, Improver, EvalReporter | Prompt quality scoring + rewrite |

**Response:**
```json
{
  "answer": "Tesla (TSLA) is trading at $247.50...",
  "steps": [
    {"step": 1, "agent": "ToolSelector", "type": "tool_routing", "content": "..."},
    {"step": 2, "agent": "ToolExecutor", "type": "tool_execution", "tool": "get_stock_price", "duration_ms": 312, "cached": false},
    {"step": 3, "agent": "Summarizer", "type": "reasoning", "content": "..."}
  ],
  "tools_used": ["get_stock_price", "get_weather"],
  "available_workflows": ["debate", "research", "smart_assistant", "smart_travel_planner", "prompt_evaluation"],
  "orchestrator_type": "autogen",
  "conversation_id": "conv_xxx"
}
```

### Workflow Examples

```bash
# Smart assistant (auto tool selection)
curl -X POST "http://127.0.0.1:8000/api/agents/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "AAPL stock price and NY weather", "workflow": "smart_assistant"}'

# Travel planner
curl -X POST "http://127.0.0.1:8000/api/agents/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Plan 3-day trip to Goa from Delhi, budget 25000 INR", "workflow": "smart_travel_planner"}'

# Debate
curl -X POST "http://127.0.0.1:8000/api/agents/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Should companies adopt remote work?", "workflow": "debate"}'

# Research (saves report to user_uploaded_files/research_reports/)
curl -X POST "http://127.0.0.1:8000/api/agents/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Impact of AI on healthcare", "workflow": "research"}'

# Prompt evaluation
curl -X POST "http://127.0.0.1:8000/api/agents/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about AI.", "workflow": "prompt_evaluation"}'

# CrewAI debate
curl -X POST "http://127.0.0.1:8000/api/agents/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Remote work policies", "orchestrator_type": "crewai", "workflow": "debate"}'
```

### Agent Tools

```bash
GET  /api/agents/tools                          # list all 21 tools
GET  /api/agents/workflows?orchestrator_type=autogen   # list workflows + tools for orchestrator
GET  /api/agents/status                         # orchestrator availability map
POST /api/agents/tools/{tool_name}/test         # test a single tool directly
GET  /api/agents/conversations/{id}/messages    # agent message history (agent_messages table)
```

**Test tool examples:**
```bash
# Single-arg tool
curl -X POST "http://127.0.0.1:8000/api/agents/tools/get_stock_price/test" \
  -H "Content-Type: application/json" \
  -d '{"input_data": "AAPL"}'

# Multi-arg tool — pass JSON string as input_data
curl -X POST "http://127.0.0.1:8000/api/agents/tools/save_text_file/test" \
  -H "Content-Type: application/json" \
  -d '{"input_data": "{\"filename\": \"out.txt\", \"content\": \"hello\"}"}'
```

**Available tools (21 total):**

| Tool | Data | API key |
|------|------|---------|
| `web_search` | DuckDuckGo (free) or SerpAPI | Optional `SERPAPI_KEY` |
| `scrape_url` | Live HTTP + BeautifulSoup | — |
| `get_stock_price` | yfinance (real) | — |
| `get_stock_history` | yfinance (real) | — |
| `get_crypto_price` | yfinance (real) | — |
| `generate_stock_chart` | yfinance + matplotlib | — |
| `generate_chart` | matplotlib | — |
| `get_weather` | OpenWeatherMap or demo fallback | Optional `OPENWEATHER_API_KEY` |
| `save_research_report` | writes markdown + JSON sidecar | — |
| `search_flights` | demo | — |
| `search_hotels` | demo | — |
| `estimate_trip_budget` | demo | — |
| `search_places` | demo (real for Goa/Jaipur/Dubai/Rome) | — |
| `search_restaurants` | demo | — |
| `generate_itinerary` | demo | — |
| `get_local_transport_info` | demo | — |
| `get_distance_between_places` | demo + city lookup | — |
| `generate_trip_summary` | demo | — |
| `get_currency_exchange` | exchangerate.host (real) | — |
| `get_geo_distance` | OpenStreetMap Nominatim (real) | — |

---

## Conversation History

Conversations are tied to `user_id`, not sessions — persistent across devices and server restarts.

**chat_type values:** `rag` · `agent` · `crew`

```bash
GET    /api/conversations                          # list conversations (filter: ?chat_type=rag)
POST   /api/conversations                          # create — body: {"chat_type": "rag", "title": "optional"}
GET    /api/conversations/{id}                     # get conversation
PUT    /api/conversations/{id}                     # rename — body: {"title": "new name"}
DELETE /api/conversations/{id}                     # soft delete
GET    /api/conversations/{id}/messages            # all messages with full RAG logging
POST   /api/conversations/{id}/restore             # restore to current session
```

```bash
# Create conversation
curl -X POST "http://127.0.0.1:8000/api/conversations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"chat_type": "rag", "title": "HR Policy Questions"}'

# Get messages
curl -X GET "http://127.0.0.1:8000/api/conversations/conv_abc123/messages" \
  -H "Authorization: Bearer $TOKEN"
```

**Message response — RAG fields (populated when `chat_type=rag`):**

| Field | Description |
|-------|-------------|
| `user_query` | Original question |
| `retrieved_context` | Retrieved docs (id, text, metadata, distance) |
| `embeddings_used` | `{model, dimensions}` |
| `llm_prompt` | Final prompt sent to LLM |
| `llm_provider` | Provider used |
| `llm_model` | Model name |
| `llm_temperature` | Temperature used |
| `llm_max_tokens` | Max tokens parameter |
| `retrieved_doc_ids` | List of doc IDs |
| `retrieval_top_k` | top_k used |
| `processing_time_ms` | Total latency |
| `error_message` | Error if query failed |

---

## Prompt Templates

Templates use JSON message arrays with variable substitution. Each template has exactly 2 messages: `system` (index 0) and `user` (index 1). History is auto-inserted between them.

```bash
POST   /api/templates               # create
GET    /api/templates               # list all
GET    /api/templates/{name}        # get by name
PUT    /api/templates/{name}        # update
DELETE /api/templates/{name}        # delete
POST   /api/templates/test/{name}   # test a template
```

```bash
# Create template
curl -X POST "http://127.0.0.1:8000/api/templates" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "pirate_template",
    "content": "[{\"role\":\"system\",\"content\":\"You are a pirate. Respond with Ahoy! and pirate language.\"},{\"role\":\"user\",\"content\":\"{user_question}\"}]",
    "prompt_variables": "user_question"
  }'

# Use template in RAG query
curl -X POST "http://127.0.0.1:8000/api/rag/local/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is PTO?", "use_llm": true, "prompt_template": "pirate_template"}'
```

**Available variables:** `{user_question}` · `{source_docs}` · `{user_role}` · `{department}` · `{user_profile_summary}` · `{max_tokens}`

`prompt_variables` — pipe-separated list of variable names. Empty string = auto-detect (backward compat).

---

## Multimodal

### Speech-to-Text
```bash
POST /api/audio/stt
```
```bash
curl -X POST "http://127.0.0.1:8000/api/audio/stt" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@question.wav" \
  -F "provider=vosk" \
  -F "conversation_id=conv_123"
```
```json
{"success": true, "data": {"text": "What is the vacation policy?", "provider": "vosk", "confidence": 0.8}, "file_path": "..."}
```
Providers: `vosk` (default, offline) · `whisper`

### Text-to-Speech
```bash
POST /api/audio/tts
```
```bash
curl -X POST "http://127.0.0.1:8000/api/audio/tts" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"text": "Here is your answer", "conversation_id": "conv_123", "provider": "pyttsx3"}'
```
Providers: `pyttsx3` (default) · `espeak`

### Emotion Detection
```bash
POST /api/audio/emotion
```
```bash
curl -X POST "http://127.0.0.1:8000/api/audio/emotion" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@voice.wav"
```
Returns: `excited` · `calm` · `positive` · `neutral`

### OCR
```bash
POST /api/vision/ocr
```
```bash
curl -X POST "http://127.0.0.1:8000/api/vision/ocr" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@document.jpg" \
  -F "provider=tesseract"
```
Providers: `tesseract` (default) · `paddleocr` · `auto`

### Image Description
```bash
POST /api/vision/describe
```

### Media File Serving
```bash
GET /api/media/{user_id}/{filename}   # RBAC — users can only access their own files
```

### Voice-to-Voice Workflow
```bash
# 1. STT
STT=$(curl -s -X POST "http://127.0.0.1:8000/api/audio/stt" \
  -H "Authorization: Bearer $TOKEN" -F "file=@q.wav" -F "conversation_id=conv_123")
Q=$(echo $STT | jq -r '.data.text')

# 2. RAG
ANS=$(curl -s -X POST "http://127.0.0.1:8000/api/rag/local/query" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"question\": \"$Q\", \"conversation_id\": \"conv_123\", \"use_llm\": true}" | jq -r '.answer')

# 3. TTS
curl -X POST "http://127.0.0.1:8000/api/audio/tts" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d "{\"text\": \"$ANS\", \"conversation_id\": \"conv_123\"}"
```

---

## Model Management

```bash
GET  /api/models/list          # list all available local GGUF models
GET  /api/models/best          # best model for current system
GET  /api/models/downloadable  # models available to download
POST /api/models/refresh       # re-scan models/ directory
```

---

## Document Cleanup & Metadata Enrichment

LLM-assisted metadata enrichment pipeline — runs on documents at ingestion time, adds semantic `summary`, `keywords`, `themes`, `entities`.

```bash
POST /api/cleanupdata                          # start pipeline (body: {"force": false})
GET  /api/cleanupdata/status                   # check progress
GET  /api/cleanupdata/preview/{document_id}    # preview original vs enriched metadata
```

```bash
# Start enrichment
curl -X POST "http://127.0.0.1:8000/api/cleanupdata" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"force": false}'

# Preview
curl -X GET "http://127.0.0.1:8000/api/cleanupdata/preview/CEO_memo_strategic_vision?version=v1" \
  -H "Authorization: Bearer $TOKEN"
```

---

## System Status

```bash
GET /api/rag/embedding/status    # embedding model info (Manager+)
GET /api/rag/sentiment/stats     # session sentiment stats (SuperAdmin)
POST /api/rag/sentiment          # analyze text sentiment (SuperAdmin)
GET /api/modules/status          # container initialization status
```

---

## RBAC & Permissions Reference

### Role → Max Sensitivity

| Role | Can read up to | Can create up to |
|------|---------------|-----------------|
| Guest | `public_internal` | — |
| Employee | `public_internal` + own `personal` | `public_internal` |
| HR | `role_confidential` + `personal` | `role_confidential` |
| Manager | `highly_confidential` | `highly_confidential` |
| SuperAdmin | `super_confidential` | `super_confidential` |

### Sensitivity Levels

| Level | Value | Access rule |
|-------|-------|-------------|
| `public_internal` | 0 | Everyone |
| `department_confidential` | 1 | Employee+ AND same department |
| `personal` | 1 | Owner OR HR+ |
| `role_confidential` | 2 | HR+ |
| `highly_confidential` | 3 | Manager+ |
| `super_confidential` | 4 | SuperAdmin only |

### RBAC Error Examples

```json
// Insufficient role
{"detail": "Your role 'Employee' (level 1) cannot create documents with sensitivity 'highly_confidential' (requires level 3+)"}

// Wrong department
{"detail": "Your role 'Employee' cannot update documents from department 'HR'. Your department is 'Engineering'."}
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| 400 | Invalid request / bad metadata |
| 401 | Missing or invalid JWT |
| 403 | RBAC denied |
| 404 | Resource not found |
| 413 | File too large (>5 MB) |
| 500 | Server error |

```json
{"detail": "Specific error message"}
```

---

## Complete Workflow Example

```bash
#!/bin/bash

# 1. Login
TOKEN=$(curl -s -X POST "http://127.0.0.1:8000/api/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | jq -r '.access_token')

# 2. (Optional) Fix query typo before submitting
SUGGESTION=$(curl -s -X POST "http://127.0.0.1:8000/api/rag/query/preprocess" \
  -H "Content-Type: application/json" \
  -d '{"query": "wht is pto polcy"}' | jq -r '.suggestion')
echo "Suggestion: $SUGGESTION"

# 3. Create conversation
CONV_ID=$(curl -s -X POST "http://127.0.0.1:8000/api/conversations" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"chat_type": "rag", "title": "PTO Questions"}' | jq -r '.id')

# 4. RAG query with corrected query
curl -s -X POST "http://127.0.0.1:8000/api/rag/local/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"$SUGGESTION\", \"conversation_id\": \"$CONV_ID\", \"use_llm\": true}" | jq '.answer'

# 5. Agent follow-up
curl -s -X POST "http://127.0.0.1:8000/api/agents/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"What is current AAPL stock?\", \"workflow\": \"smart_assistant\", \"conversation_id\": \"$CONV_ID\"}" | jq '.answer'

# 6. View conversation history
curl -s "http://127.0.0.1:8000/api/conversations/$CONV_ID/messages" \
  -H "Authorization: Bearer $TOKEN" | jq '[.[] | {speaker, content}]'
```

---

**Last Updated:** 2025-06-15
**API Version:** 1.0.0
