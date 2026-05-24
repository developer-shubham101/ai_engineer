# Technology Stack

## Language & Runtime
- **Python 3.8+** (uses `from __future__ import annotations` for forward refs)
- **Windows** primary dev environment (`.bat` scripts, Windows paths in settings)

## Web Framework
- **FastAPI** — async HTTP framework, automatic OpenAPI/Swagger at `/docs`
- **Uvicorn** (with `[standard]` extras) — ASGI server
- **Pydantic** — request/response validation and settings models
- **python-multipart** — file upload support

## AI / LLM Layer
| Library | Purpose |
|---|---|
| `llama-cpp-python` | Local GGUF model inference (Mistral, Phi-2, Llama, Gemma) |
| `openai` | OpenAI GPT API client |
| `langchain`, `langchain-core`, `langchain-community` | Prompt chains, template management |
| `langchain-google-genai` | Google Gemini integration |
| `langchain-text-splitters` | Document chunking |
| `crewai` | Multi-agent crew workflows |
| `autogen-core`, `autogen-agentchat`, `autogen-ext[openai]` | AutoGen agent framework |
| `huggingface_hub` | HF Inference API access |
| `transformers` | Local transformer models |
| `torch` | PyTorch backend for local models |

## Vector Store & Retrieval
| Library | Purpose |
|---|---|
| `chromadb` | ChromaDB vector store (persistent) |
| `faiss-cpu` | FAISS vector store (default, env-switchable) |
| `sentence-transformers` | Embedding generation (BGE, MiniLM, MPNet, E5) |
| `rank-bm25` | BM25 keyword retrieval |
| `pyspellchecker` | Query spell correction |
| `tiktoken` | Token counting for OpenAI models |

## Authentication & Security
- **PyJWT** — JWT token creation and verification (HS256)
- **passlib[bcrypt]** — Password hashing
- **SQLite** — User, session, conversation, and version databases

## Multimodal
| Library | Purpose |
|---|---|
| `vosk` | Offline speech-to-text |
| `openai-whisper` | Cloud/local STT |
| `pyttsx3` | Text-to-speech |
| `librosa`, `soundfile` | Audio processing |
| `pytesseract` | OCR (requires Tesseract binary) |
| `Pillow` | Image processing |
| `paddlepaddle`, `paddleocr` | Advanced OCR |
| `opencv-python` | Computer vision |
| `ultralytics` | YOLO object detection |

## Data & Utilities
- **python-dotenv** — `.env` file loading
- **PyYAML** — CrewAI config parsing (`crew_config/*.yaml`)
- **PyPDF2** — PDF text extraction
- **beautifulsoup4** — HTML parsing
- **markdown** — Markdown rendering
- **rich** — Terminal output formatting
- **yfinance** — Financial data for agent tools
- **duckduckgo-search** — Web search for agent tools

## Testing
- **pytest** — Test runner
- **pytest-asyncio** — Async test support
- Test suite in `test_module/` with `conftest.py` fixtures

## Databases (SQLite files in `database/`)
| File | Purpose |
|---|---|
| `users.db` | User accounts and credentials |
| `sessions.db` | JWT session tracking |
| `conversations.db` | Conversation history |
| `document_versions.db` | Document version history |

## Vector Store Selection
Controlled by `VECTOR_STORE_TYPE` environment variable:
- `faiss` (default) → `FaissVectorStore`
- anything else → `ChromaVectorStore` (persists to `chroma_storage/`)

## Embedding Models (configurable via `EMBEDDING_MODEL_KEY`)
| Key | Model | Dims | Notes |
|---|---|---|---|
| `all-MiniLM-L6-v2` | all-MiniLM-L6-v2 | 384 | Fastest |
| `bge-small-en-v1.5` | BAAI/bge-small-en-v1.5 | 384 | Default, fast+accurate |
| `bge-base-en-v1.5` | BAAI/bge-base-en-v1.5 | 768 | Best CPU accuracy |
| `e5-base-v2` | intfloat/e5-base-v2 | 768 | Multi-domain |
| `all-mpnet-base-v2` | sentence-transformers/all-mpnet-base-v2 | 768 | Production classic |

## Environment Variables (`.env.example`)
```bash
# LLM Providers
OPENAI_API_KEY=
GOOGLE_API_KEY=
HUGGINGFACE_API_TOKEN=

# Custom LLM endpoints
COLABLLM_BASE_URL=
COLABLLM_API_KEY=
CUSTOMLLM_BASE_URL=
LLAMASERVER_BASE_URL=http://192.168.1.10:8080/v1
LLAMASERVER_MODEL_NAME=mistral-7b-instruct-v0.2

# Server
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Models
DEFAULT_MODEL_NAME=mistral-7b-instruct-v0.2.Q3_K_M.gguf
EMBEDDING_MODEL_KEY=bge-small-en-v1.5
EMBEDDING_MODEL_NAME=BAAI/bge-small-en-v1.5

# Vector store
VECTOR_STORE_TYPE=faiss

# Auth
JWT_SECRET_KEY=your-secret-key-change-in-production
JWT_EXPIRATION_DAYS=1
```

## Development Commands
```bash
# Install dependencies
pip install -r requirements.txt

# Start server
python -m app.main
# or
python run_app.py

# Run tests
python run_tests.py
pytest test_module/

# Seed sample data
python scripts/seed_examples.py

# Download embedding models
python scripts/download_embeddings_models.py

# Download HF model
python scripts/download_hf_model.py

# Train sentiment model
python scripts/train_sentiment.py

# Docker
docker-compose up
```

## Token Limits (from `constants.py`)
```python
MAX_PROMPT_TOKENS = 4096
MAX_CONTEXT_TOKENS = 2048
MAX_SYSTEM_TOKENS = 80
MAX_HISTORY_TURNS = 5
DEFAULT_TOP_K = 3
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.1
```

## API Prefixes
```python
API_PREFIX = "/api"
RAG_PREFIX = "/api/rag"
AUTH_PREFIX = "/api/auth"
MODELS_PREFIX = "/api/models"
```
