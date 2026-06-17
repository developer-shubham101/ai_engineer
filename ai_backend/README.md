# 🚀 Multi-Provider Enterprise RAG System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> **Enterprise-grade RAG system with multi-provider LLM support, advanced RBAC, and offline-first architecture**
>
> 🎓 **Educational Reference Implementation**: This backend serves as a comprehensive study guide for building scalable AI systems.

A production-ready **Retrieval-Augmented Generation (RAG) system** that works with multiple LLM providers through a unified API. Supports both **offline operation** with local models and **cloud integration** with major AI providers.

---

## ✨ Key Features

- 🤖 **Multi-Provider RAG** — Local (GGUF), OpenAI GPT, Google Gemini, Hugging Face, CustomLLM, LlamaServer
- 🔒 **Enterprise RBAC** — Role hierarchy with sensitivity levels and department restrictions
- 📚 **Document Versioning** — Non-destructive updates with full version history
- 💬 **Persistent Conversations** — SQLite-backed history with cross-device access
- 🚀 **Offline-First** — Fully functional without internet using local GGUF models
- 🤖 **4 Agent Orchestrators** — AutoGen, Custom, MCP, CrewAI via unified `/api/agents/query`
- 🎭 **5 Agent Workflows** — debate, research, smart_assistant, smart_travel_planner, prompt_evaluation
- 🔍 **Hybrid Retrieval** — BM25 + vector search with cross-encoder reranking
- 🧩 **Paragraph-Aware Chunking** — Semantic boundary-respecting document chunking
- 🎙️ **Multimodal** — Speech-to-Text, Text-to-Speech, OCR, emotion detection
- 🛡️ **JWT Auth** — HS256 tokens with audit logging
- ⚡ **Prompt Optimization** — Token budgeting, context truncation, template system
- 🔌 **MCP Support** — Model Context Protocol orchestrator

---

## 🏗️ Architecture

```
HTTP Request
  → FastAPI Router
  → JWT Auth (optional)
  → RBAC Check
  → Container (DI)
  → Service (RAG / Agent / CrewAI)
  → Response
```

### Supported RAG Providers

| Provider | Endpoint | Status |
|----------|----------|--------|
| Local GGUF | `local` | ✅ Offline |
| OpenAI GPT | `gpt`, `openai` | ✅ API |
| Google Gemini | `google` | ✅ API |
| Hugging Face | `huggingface`, `hf` | ✅ API |
| CustomLLM | `customllm` | ✅ API (preferred for 3rd-party) |
| LlamaServer | `llamaserver` | ✅ Local server |

### Agent Orchestrators

| Type | `orchestrator_type` | Workflows |
|------|---------------------|-----------|
| AutoGen v0.4 | `autogen` | All 5 |
| Custom (LLM-loop) | `custom` | debate, research, smart_assistant, smart_travel_planner |
| MCP | `mcp` | smart_assistant |
| CrewAI | `crewai` | debate, research, smart_travel_planner |

### Agent Workflows

| Workflow | Agents | Description |
|----------|--------|-------------|
| `debate` | Advocate, Critic, Moderator | Multi-perspective debate |
| `research` | Planner, Researcher, Verifier, Analyst, Evaluator, ReportWriter | 6-agent research pipeline |
| `smart_assistant` | ToolSelector, ToolExecutor, Summarizer | Auto tool selection + execution |
| `smart_travel_planner` | TravelToolSelector, ToolExecutor, TravelPlanner | Intent-driven travel planning |
| `prompt_evaluation` | PromptParser, CriteriaJudge, Improver, EvalReporter | Prompt quality scoring + rewrite |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- 8GB+ RAM (for local models)

### Installation

```bash
git clone https://github.com/developer-shubham101/ai_engineer.git
cd ai_engineer/ai_backend
pip install -r requirements.txt
cp .env.example .env   # edit with your API keys
python ./scripts/download_embeddings_models.py # Download embeddings models
```

> **💡 Local LLM (Optional):** If you want to use a local model instead of cloud APIs:
> 1. Download the llama.cpp binary from **[github.com/ggml-org/llama.cpp/releases](https://github.com/ggml-org/llama.cpp/releases)** (e.g. `llama-b7445-bin-win-cpu-x64.zip`)
> 2. Start the server **before** running the app — see **[Run Local LLM Server](documents/llm_cpp/run_local_llm_cpp.md)** for commands
> 3. Set the URLs in your `.env`: `LLAMASERVER_BASE_URL=http://localhost:8080/v1` and `CREW_BASE_URL=http://localhost:8080`
>
> If you prefer cloud providers (OpenAI, Google Gemini, Hugging Face), just add the relevant API keys to your `.env` and skip this step.

```bash
python -m app.main
```

### First Query

```bash
# RAG query (no auth needed)
curl -X POST "http://localhost:8000/api/rag/local/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the company policies?", "use_llm": true}'

# Agent query
curl -X POST "http://localhost:8000/api/agents/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Tesla stock price?", "orchestrator_type": "autogen", "workflow": "smart_assistant"}'

# Prompt evaluation
curl -X POST "http://localhost:8000/api/agents/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Tell me about AI.", "orchestrator_type": "autogen", "workflow": "prompt_evaluation"}'
```

---

## 🔧 Configuration

### Environment Variables

```bash
# Cloud RAG providers (optional)
OPENAI_API_KEY=<your_openai_key>
GOOGLE_API_KEY=<your_google_key>
HUGGINGFACE_API_TOKEN=<your_hf_token>

# Custom / third-party LLM
CUSTOMLLM_BASE_URL=http://localhost:8080
CUSTOMLLM_API_KEY=<your_key>

# LlamaServer (local OpenAI-compatible server)
LLAMASERVER_BASE_URL=http://127.0.0.1:8080/v1
LLAMASERVER_MODEL_NAME=mistral-7b-instruct-v0.2

# CrewAI LLM (llama-server endpoint)
CREW_BASE_URL=http://localhost:8080

# Agent tools (optional upgrades)
OPENWEATHER_API_KEY=<your_key>   # real weather data
SERPAPI_KEY=<your_key>           # upgrade web_search from DuckDuckGo to SerpAPI

# Vector store: faiss (default) or chroma
VECTOR_STORE_TYPE=faiss

# Embedding model key
EMBEDDING_MODEL_KEY=bge-small-en-v1.5

# Auth
JWT_SECRET_KEY=<change-in-production>
JWT_EXPIRATION_DAYS=1

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false
```

### Local Models

Place GGUF files in `models/`:

```
models/
├── mistral-7b-instruct-v0.2.Q3_K_M.gguf
├── phi-2-q4_k_m.gguf
├── llama-3.2-1b-instruct-q4_k_m.gguf
└── gemma-2b-it-q4_k_m.gguf
```

---

## 🔐 Security & RBAC

### Role Hierarchy

```
SuperAdmin (4) → Manager (3) → HR (2) → Employee (1) → Guest (0)
```

### Sensitivity Levels

| Level | Value | Access |
|-------|-------|--------|
| `super_confidential` | 4 | SuperAdmin only |
| `highly_confidential` | 3 | Manager+ |
| `role_confidential` | 2 | HR+ |
| `department_confidential` | 1 | Employee+ (same dept) |
| `public_internal` | 0 | Everyone |
| `personal` | 1 | Owner + HR+ |

---

## 🔌 API Reference

### RAG Queries

```http
POST /api/rag/{provider}/query
```

```json
{
  "question": "What are the leave policies?",
  "conversation_id": "conv_xxx",
  "top_k": 3,
  "use_llm": true,
  "max_tokens": 256,
  "temperature": 0.1,
  "debug": false
}
```

### Agent Workflows

```http
POST /api/agents/query
```

```json
{
  "question": "Plan a 3-day trip to Goa",
  "orchestrator_type": "autogen",
  "workflow": "smart_travel_planner",
  "tools": [],
  "max_steps": 5
}
```

```http
GET /api/agents/workflows?orchestrator_type=autogen
GET /api/agents/tools
GET /api/agents/status
POST /api/agents/tools/{tool_name}/test
```

### Authentication

```http
POST /api/auth/token
```

```json
{"username": "<username>", "password": "<password>"}
```

### Conversations

```http
GET  /api/conversations
POST /api/conversations
GET  /api/conversations/{id}/messages
PUT  /api/conversations/{id}
DELETE /api/conversations/{id}
```

### Document Management

```http
POST /api/rag/documents/add
POST /api/rag/documents/add-file
POST /api/rag/documents/update
POST /api/rag/documents/seed
GET  /api/rag/documents/list
GET  /api/rag/documents/{id}/versions
```

### Prompt Templates

```http
POST   /api/templates
GET    /api/templates
GET    /api/templates/{name}
PUT    /api/templates/{name}
DELETE /api/templates/{name}
POST   /api/templates/test/{name}
```

### Multimodal

```http
POST /api/audio/stt      # Speech-to-Text (Vosk / Whisper)
POST /api/audio/tts      # Text-to-Speech (pyttsx3)
POST /api/audio/emotion  # Emotion detection
POST /api/vision/ocr     # OCR (Tesseract / PaddleOCR)
POST /api/vision/describe
GET  /api/media/{user_id}/{filename}
```

### Metadata Enrichment

```http
POST /api/cleanupdata
GET  /api/cleanupdata/status
GET  /api/cleanupdata/preview/{document_id}
```

---

## 🛠️ Development

### Project Structure

```
ai_backend/
├── app/
│   ├── modules/
│   │   ├── agents/
│   │   │   ├── function_tools/       # 21 callable tools
│   │   │   ├── orchestrators/
│   │   │   │   ├── autogen/          # AutoGen v0.4 orchestrator + workflows/
│   │   │   │   ├── custom/           # LLM-loop orchestrator + workflows/
│   │   │   │   ├── mcp/              # MCP orchestrator
│   │   │   │   ├── crewai/           # CrewAI orchestrator + travel_workflow
│   │   │   │   └── utils/            # Shared: tool_registry, plan_normalizer, step_utils...
│   │   │   ├── interfaces.py
│   │   │   └── factories.py
│   │   ├── auth/                     # JWT, RBAC, user/session managers
│   │   ├── config/                   # Settings, constants, model configs
│   │   ├── conversation/             # SQLite conversation history
│   │   ├── core/                     # DocumentManager, VersionManager, chunking
│   │   ├── llm/                      # RAGOrchestrator, providers, prompt system
│   │   ├── multimodal/               # STT, TTS, OCR, emotion
│   │   ├── vector_db/                # FAISS/Chroma, BM25, reranker, embeddings
│   │   └── integration.py            # DI Container
│   ├── api_routes_*.py               # Route files (no api_routes_crew.py)
│   └── main.py
├── crew_config/
│   ├── agents.yaml                   # CrewAI agent definitions
│   └── tasks.yaml                    # CrewAI task definitions
├── data/                             # Sample documents
│   └── cleaned/                      # LLM-enriched documents
├── models/                           # Local GGUF models
├── database/                         # SQLite databases
├── requirements.txt
├── requirements_agents.txt           # Agent-specific deps
├── requirements_mcp.txt              # MCP deps
└── requirements_multimodal.txt       # Multimodal deps
```

### Adding a New Agent Workflow

1. Create `app/modules/agents/orchestrators/autogen/workflows/my_workflow.py`
2. Export from `workflows/__init__.py`
3. Add to `AutoGenOrchestrator.AVAILABLE_WORKFLOWS` and `WORKFLOW_REGISTRY`
4. Add dispatcher `_run_my_workflow()` in `autogen_orchestrator.py`
5. Mirror in `custom/workflows/` and `crewai/orchestrator.py` + YAML configs

### Adding a New RAG Provider

1. Create provider class in `app/modules/llm/providers/`
2. Register in `provider_factory.py`
3. Add to `VALID_PROVIDERS` in `constants.py`

### Running Tests

```bash
pytest test_module/ -v
python validate_container_full.py
```

---

## 📚 Documentation

- **[API Docs](http://localhost:8000/docs)** — Interactive Swagger UI
- **[APP_CONTEXT.md](APP_CONTEXT.md)** — Complete technical architecture reference
- **[AUTO_GEN.md](AUTO_GEN.md)** — AutoGen orchestrator deep-dive
- **[Run Local LLM Server](documents/llm_cpp/run_local_llm_cpp.md)** — llama.cpp server setup & commands

---

## 🙏 Acknowledgments

- **FastAPI** — Web framework
- **ChromaDB / FAISS** — Vector stores
- **Sentence Transformers** — Embedding models
- **llama-cpp-python** — Local LLM inference
- **AutoGen v0.4** — Multi-agent framework
- **CrewAI** — Crew-based multi-agent workflows
- **OpenAI, Google, Hugging Face** — Cloud AI providers

---

<div align="center">

**Built with ❤️ for the AI community**

[Report Bug](https://github.com/developer-shubham101/ai_engineer/issues) • [Request Feature](https://github.com/developer-shubham101/ai_engineer/issues)

</div>
