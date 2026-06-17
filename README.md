# 🚀 Multi-Provider Enterprise RAG System

> A **comprehensive reference implementation** for AI Engineers learning to build production-grade RAG systems with multi-provider LLM support, enterprise RBAC, and a modern React UI.

---

## 📦 Repository Structure

| Module | Description | Readme |
|--------|-------------|--------|
| `ai_backend/` | FastAPI backend — RAG, Agents, Auth, RBAC, Conversations | [ai_backend/README.md](ai_backend/README.md) |
| `front_end_react/` | React + Vite frontend — Chat UI, Document Management, Admin | [front_end_react/README.md](front_end_react/README.md) |
| `mcp/` | Standalone MCP Tools Server — Weather, Stock, Web, Travel | [mcp/README.md](mcp/README.md) |
| `lora_traning/` | LoRA fine-tuning pipeline for Llama 3.2 1B | `lora_traning/USAGE_GUIDE.md` |
| `vector_databases/` | Standalone demos — ChromaDB, FAISS, LanceDB, SQLite | — |

---

## ✨ Highlights

- 🤖 **Multi-Provider RAG** — OpenAI GPT, Google Gemini, Hugging Face, local GGUF, LlamaServer, CustomLLM
- 🔒 **Enterprise RBAC** — Role hierarchy (`Guest → SuperAdmin`), sensitivity levels, department restrictions
- 💬 **Persistent Conversations** — SQLite-backed history with session isolation
- 🤖 **4 Agent Orchestrators** — AutoGen, Custom, MCP, CrewAI via unified `/api/agents/query`
- 🔍 **Hybrid Retrieval** — BM25 + vector search with cross-encoder reranking
- 🎙️ **Multimodal** — Speech-to-Text, Text-to-Speech, OCR, emotion detection
- 🔌 **MCP Server** — Standalone tool server compatible with Claude Desktop and any MCP client

---

## 🚀 Getting Started

### Option 1 — Docker (recommended)

Runs the backend and frontend together with a single command.

```bash
git clone https://github.com/developer-shubham101/ai_engineer.git
cd ai_engineer
docker compose up --build
```

| Service | URL |
|---------|-----|
| Backend API | http://localhost:8000 |
| Frontend UI | http://localhost:3000 |
| Swagger Docs | http://localhost:8000/docs |

To point the frontend at a custom backend URL:

```bash
VITE_API_URL=https://api.yourdomain.com docker compose up --build
```

---

### Option 2 — Local Setup

```bash
git clone https://github.com/developer-shubham101/ai_engineer.git
cd ai_engineer
```

#### Backend

> See [ai_backend/README.md](ai_backend/README.md) for full configuration, provider setup, agent workflows, and API reference.

```bash
cd ai_backend
pip install -r requirements.txt
cp .env.example .env          # add your API keys
python scripts/download_embeddings_models.py
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Backend available at **http://localhost:8000**
Interactive API docs at **http://localhost:8000/docs**

---

#### Frontend

> See [front_end_react/README.md](front_end_react/README.md) for component structure and UI patterns.

```bash
cd front_end_react
npm install
npm run dev
```

Frontend available at **http://localhost:3000**

> The API URL defaults to `http://localhost:8000`. To change it, set `VITE_API_URL` in a `.env` file inside `front_end_react/`.

---

#### MCP Tools Server

> See [mcp/README.md](mcp/README.md) for tool list, Claude Desktop integration, and inspector usage.

```bash
cd mcp
pip install -r requirements.txt
cp .env.example .env          # optional: add OPENWEATHER_API_KEY, SERPAPI_KEY
python server.py              # stdio transport
```

Or run the interactive client REPL:

```bash
python client.py
```

---

## 🔧 Environment Variables

Copy `.env.example` to `.env` inside `ai_backend/` and fill in the keys you need:

```bash
cp ai_backend/.env.example ai_backend/.env
```

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI GPT provider |
| `GOOGLE_API_KEY` | Google Gemini provider |
| `HUGGINGFACE_API_TOKEN` | Hugging Face Inference API |
| `CUSTOMLLM_BASE_URL` | Any OpenAI-compatible 3rd-party LLM |
| `JWT_SECRET_KEY` | Auth token signing — **change in production** |
| `VECTOR_STORE_TYPE` | `faiss` (default) or `chroma` |

Full variable reference → [ai_backend/README.md](ai_backend/README.md#-configuration)

---

## 📖 API Documentation

- **Interactive (Swagger UI)** → http://localhost:8000/docs
- **Full reference** → [ai_backend/documents/API_DOCUMENTATION.md](ai_backend/documents/API_DOCUMENTATION.md)

---

## 🔒 RBAC & Security

Role hierarchy: `Guest (0) < Employee (1) < HR (2) < Manager (3) < SuperAdmin (4)`

Sensitivity levels: `public_internal` → `department_confidential` → `role_confidential` → `highly_confidential` → `super_confidential`

Full RBAC docs → [ai_backend/README.md](ai_backend/README.md#-security--rbac)

---

## 🤝 Contributing

Contributions are welcome! Fork the repo, create a feature branch, and open a pull request.

- 🐛 [Report a Bug](https://github.com/developer-shubham101/ai_engineer/issues)
- 💡 [Request a Feature](https://github.com/developer-shubham101/ai_engineer/issues)

---

## 🧑💻 Author

**Shubham Sharma**
[GitHub](https://github.com/developer-shubham101) • [Repository](https://github.com/developer-shubham101/ai_engineer)
