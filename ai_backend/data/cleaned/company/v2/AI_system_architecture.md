Saarthi Infotech: Multi-Provider Enterprise RAG System Architecture
Document Version: 2.0
Last Updated: January 15, 2025
Owner: Engineering Team
Classification: Role Confidential
System Overview
Saarthi Infotech has developed a versatile enterprise RAG (Retrieval-Augmented Generation) system supporting multiple LLM providers through a unified architecture. The system operates both offline with local models and integrates with cloud APIs.
Supported Providers
Local Models: Mistral-7B-Instruct-v0.2.Q3_K_M.gguf (CPU-only, offline)
Google Gemini API: gemini-2.5-flash, gemini-2.5-pro
OpenAI GPT API: gpt-3.5-turbo, gpt-4
Hugging Face Inference API: Various models
Core Components
Local embeddings (MiniLM) - shared vector space across all providers
Local Chroma vector DB - unified document storage
FastAPI backend - single API interface
Flexible RBAC - level-based + role overrides
Session-aware Support Chat System - persistent conversations
Architecture Flow
User → FastAPI → RAG Pipeline → RBAC Filter → LLM Provider → Response
↓                      ↓
Chroma Vector DB        [Local|Google|GPT|HF]
↓
Local Embeddings (MiniLM)
RBAC Implementation
Role Hierarchy
SuperAdmin (4) - Full system access
Manager (3) - Management + below
HR (2) - HR functions + below
Employee (1) - Standard access + public
Guest (0) - Public content only
Sensitivity Levels
public_internal (0) – Everyone
department_confidential (1) – Employee+ in same department
role_confidential (2) – HR+ level
highly_confidential (3) – Manager+ level
super_confidential (4) – SuperAdmin only
personal (1) – Owner + HR+ level
API Endpoints
Multi-Provider RAG
POST /api/rag/local/query - Local Mistral-7B
POST /api/rag/google/query - Google Gemini
POST /api/rag/gpt/query - OpenAI GPT
POST /api/rag/huggingface/query - Hugging Face
Document Management
POST /api/rag/documents/add - Add document (JSON)
POST /api/rag/documents/seed - Seed default data
POST /api/rag/documents/update - Update with versioning
Security Considerations
All document chunks are filtered through RBAC before being sent to any LLM provider. This ensures consistent security across local and cloud models.
Performance Metrics
Local model: ~2-5 seconds response time
Cloud APIs: ~1-3 seconds response time
Vector search: <100ms
RBAC filtering: <50ms per document chunk