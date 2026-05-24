# Product Overview

## Purpose
Multi-Provider Enterprise RAG (Retrieval-Augmented Generation) system — a production-ready, educational reference implementation for building scalable AI backends. Designed to teach and demonstrate RAG architecture patterns with real enterprise features.

## Value Proposition
- Single unified API across multiple LLM providers (local + cloud)
- Offline-first: fully functional without internet using local GGUF models
- Enterprise-grade security with RBAC, JWT auth, and audit logging
- Modular, extensible architecture for learning and production use

## Key Features

### LLM Providers
- **Local**: Mistral-7B, Phi-2, Llama-3.2, Gemma-2B via llama-cpp-python (GGUF format)
- **Cloud**: OpenAI GPT-3.5/4, Google Gemini-2.5-Flash/Pro, Hugging Face Inference API
- **Custom**: ColabLLM / CustomLLM (any OpenAI-compatible endpoint), LlamaServer

### RAG & Retrieval
- Vector search via ChromaDB or FAISS (switchable via `VECTOR_STORE_TYPE` env var)
- BM25 keyword search with hybrid retrieval fusion
- Cross-encoder reranking for improved result quality
- Query preprocessing and spell correction
- Smart token budgeting and context truncation

### Agent Frameworks
- AutoGen multi-agent orchestration
- CrewAI workflows: `debate` (Advocate/Critic/Moderator) and `research` (Researcher/Analyst/Synthesizer)
- Modular agent tools (web search, RAG lookup, financial data via yfinance)

### Security & Access Control
- JWT authentication (HS256, configurable expiry)
- Role hierarchy: Guest(0) → Employee(1) → HR(2) → Manager(3) → SuperAdmin(4)
- Document-level sensitivity filtering: `public_internal`, `department_confidential`, `role_confidential`, `highly_confidential`, `super_confidential`, `personal`
- Department-based access restrictions
- Full audit logging of access attempts

### Document Management
- Non-destructive versioning (new versions, never overwrites)
- Metadata-driven access control per document chunk
- Multi-format ingestion: `.txt`, `.md`, `.html`, `.json`, `.csv`, PDF
- Automated metadata generation via LLM

### Multimodal
- Speech-to-text: Vosk (offline), OpenAI Whisper
- Text-to-speech: pyttsx3
- Vision/OCR: Tesseract, PaddleOCR, OpenCV, YOLO (Ultralytics)
- Emotion detection providers

### Conversation & Sessions
- Persistent conversation history (SQLite)
- Session management with context window
- Prompt template system with LangChain integration

## Target Users
- **AI/ML Engineers** learning RAG system design
- **Enterprise developers** building internal knowledge bases
- **Researchers** experimenting with LLM providers and retrieval strategies
- **Teams** needing role-gated document Q&A systems

## Use Cases
- Enterprise knowledge management with role-based access
- Document Q&A over company policies, HR handbooks, technical docs
- Multi-agent research and debate workflows
- Prototyping AI-powered applications with provider flexibility
