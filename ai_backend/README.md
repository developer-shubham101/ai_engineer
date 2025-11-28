# 🚀 **Multi-Provider Enterprise RAG System**

**AI-First Project Summary — For LLM Understanding**

This project is a **versatile enterprise RAG system** designed for **learning, experimentation, and testing** with support for multiple LLM providers through a unified architecture. The system can run **fully offline** with local models or integrate with **cloud APIs** through a common interface.

Supported providers:

* **Local Models**: Mistral-7B-Instruct-v0.2.Q3_K_M.gguf (CPU-only, offline)
* **Google Gemini API**: gemini-2.5-flash, gemini-2.5-pro
* **OpenAI GPT API**: gpt-3.5-turbo, gpt-4
* **Hugging Face Inference API**: Various models

Common components across all providers:

* **Local embeddings (BGE-Small-EN-v1.5)** - shared vector space
* **Local Chroma vector DB** - unified document storage
* **FastAPI backend** - single API interface
* **Flexible RBAC (Role-Based Access Control)** - level-based + role overrides
* **Session-aware Support Chat System (SQLite)** - persistent conversations

This README describes the project **from an AI / system-architecture perspective**, so any LLM can easily understand how the system works and give accurate help.

---

# 🧠 **1. High-Level Purpose**

This project simulates a **real enterprise AI assistant** inside a fictional company *Saarthi Infotech Pvt. Ltd.*

The system can:

### ✅ Answer role-specific questions

(Policies, workflows, HR, Finance, IT, Legal, etc.)

### ✅ Enforce flexible RBAC

**Level-based hierarchy**: SuperAdmin (4) → Manager (3) → HR (2) → Employee (1) → Guest (0)

**Six sensitivity levels**: `public_internal`, `department_confidential`, `role_confidential`, `highly_confidential`, `super_confidential`, `personal`

**Role overrides**: `allowed_roles` can bypass hierarchy (e.g., Admin+Employee only, blocks Manager/HR)

### ✅ Perform RAG queries locally

Using BGE embeddings + Chroma

### ✅ Support multiple LLM providers

Local (Mistral-7B-Instruct GGUF), Google Gemini, OpenAI GPT, Hugging Face

### ✅ Maintain multi-turn support chat sessions

Stored in SQLite for short-term memory

### ✅ Provide natural-language AI responses

Built from allowed document chunks only

The system supports **offline-first** operation with local models, plus **cloud integration** for enhanced capabilities, designed for **learning and testing**.

---

# 🏗 **2. System Architecture Overview**

```
User → FastAPI → RAG Pipeline → RBAC Filter → LLM Provider → Response
                          ↓                      ↓
                   Chroma Vector DB        [Local|Google|GPT|HF]
                          ↓
                Local Embeddings (MiniLM)
```

### Components:

| Component                    | Purpose                                                     |
| ---------------------------- | ----------------------------------------------------------- |
| **FastAPI Server**           | Provides REST endpoints for query, add, seed, chat sessions |
| **Base RAG Service**         | Common functionality: retrieval, RBAC filtering, sessions   |
| **Provider Services**        | Local (Mistral), Google (Gemini), GPT, Hugging Face        |
| **ChromaDB**                 | Stores vector embeddings + metadata (`chroma_storage/`)     |
| **SQLite Databases**         | User auth, chat sessions, document versions (`database/`)   |
| **Auth Layer**               | JWT-based authentication with role management               |
| **Role & Department System** | Controls document visibility across all providers           |

---

# 🔐 **3. Flexible RBAC System**

### **Role Hierarchy** (Level-based access)

* **SuperAdmin (4)** - Full system access
* **Manager (3)** - Management + below
* **HR (2)** - HR functions + below  
* **Employee (1)** - Standard access + public
* **Guest/PublicUser (0)** - Public content only

### **Sensitivity Levels** (Required access level)

* `public_internal` (0) – Everyone
* `department_confidential` (1) – Employee+ in same department
* `role_confidential` (2) – HR+ level
* `highly_confidential` (3) – Manager+ level
* `super_confidential` (4) – SuperAdmin only
* `personal` (1) – Owner + HR+ level

### **Advanced Features**

* **Role Override**: `allowed_roles` bypasses hierarchy
* **Department Restrictions**: Users below HR level can only update their department
* **Level Validation**: Users can only create documents at their level or below

### **Examples**

```json
// Only Admin + Employee access (blocks Manager/HR)
{"sensitivity": "highly_confidential", "allowed_roles": ["SuperAdmin", "Employee"]}

// Department-specific with override
{"sensitivity": "department_confidential", "department": "HR", "allowed_roles": ["SuperAdmin", "HR"]}
```

This metadata is stored per document chunk and enforced during retrieval and creation.

---

# 🆕 **Recent Updates (Latest Commits)**

## Prompt Optimization & Debug System
- **Token Budgeting**: Dynamic allocation between system instructions (60-80 tokens), context (65%), and user query
- **Smart Context Truncation**: Automatic truncation when content exceeds model limits with head/tail preservation
- **Debug Exposure**: New `final_prompt` field in API responses for optimization analysis
- **Performance Metrics**: Complete token usage tracking and efficiency monitoring

## Enhanced Document Management
- **Versioned Company Data**: Structured v1/v2 document hierarchy with comprehensive metadata
- **BGE Embeddings**: Upgraded to 'bge-small-en-v1.5' for improved semantic understanding
- **Archive System**: Clean separation of current vs archived documents

## Improved Logging & Monitoring
- **RBAC Audit Trails**: Detailed logging of all access control decisions
- **LLM Interaction Logs**: Complete prompt/response tracking for debugging
- **Performance Monitoring**: Response times, token efficiency, and usage patterns

---

# 📚 **4. RAG Flow (AI-Focused Explanation)**

### 1. User asks a question

→ API receives: question, role, department, optional session.

### 2. Query text is embedded locally

Using **SentenceTransformers / BGE-Small-EN-v1.5**.

### 3. Chroma returns top-k chunks

But these chunks may include restricted content.

### 4. **Flexible RBAC Filtering**

Each chunk is checked by:

1. **Personal documents**: Owner or HR+ level
2. **Role overrides**: `allowed_roles` bypasses hierarchy
3. **Department matching**: For `department_confidential`
4. **Level-based access**: User level ≥ required level

```python
user_level = ROLE_LEVELS.get(user_role, 0)
required_level = SENSITIVITY_LEVELS.get(sensitivity, 0)
access_granted = user_level >= required_level
```

Unauthorized chunks:

* Are removed with audit logging
* Public summaries may be shown as fallback
* Count of filtered items is recorded

### 5. AI Prompt is built (optional session prefix)

Including:

* Support category (HR/IT/etc)
* Last 5 messages (history)
* User role / department context
* Allowed chunks only

### 6. Provider-specific LLM generates answer

Local Mistral, Google Gemini, OpenAI GPT, or Hugging Face - using only the allowed visible context.

---

# 💬 **5. Support Chat Subsystem**

A lightweight chat system stores:

* Session metadata
* User & assistant messages
* Timestamped history
* Conversation memory (last 5 turns)

This is used to build a dynamic prefix for the LLM.

### Example generated prompt prefix:

```
You are an HR support assistant.
User role: Employee, Department: Engineering
Conversation history:
[2025-01-10] USER: How do I apply for leave?
[2025-01-10] ASSISTANT: …
```

This allows local multi-turn AI behavior **without external APIs**.

---

# 🗂 **6. Document Ingestion**

### Two methods:

1. `/api/rag/documents/add` – JSON text
2. `/api/rag/documents/add-file` – Upload `.txt`

### Metadata includes:

```
department
sensitivity
allowed_roles
owner_id
public_summary
ingested_at
ingested_by
```

### Chunking:

* 512 chars
* 64 overlap
* Embedded locally

Chunks are then stored in ChromaDB.

---

# 🧪 **7. Multi-Provider LLM Support**

### **Local Provider** (Offline)
**mistral-7b-instruct-v0.2.Q3_K_M.gguf** via llama.cpp:
* CPU-only, no internet required
* Lazy loaded at first query
* Auto-detected from `/models/*.gguf`

### **Cloud Providers** (API-based)
* **Google Gemini**: gemini-2.5-flash (default), gemini-2.5-pro
* **OpenAI GPT**: gpt-3.5-turbo (default), gpt-4
* **Hugging Face**: Various models via Inference API

### **Environment Variables**
```bash
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key  
HUGGINGFACE_API_TOKEN=your_hf_token
```

---

# 🔌 **8. API Endpoints (Simplified)**

### **Multi-Provider RAG**

* `POST /api/rag/local/query` - Local Mistral-7B
* `POST /api/rag/google/query` - Google Gemini
* `POST /api/rag/gpt/query` - OpenAI GPT
* `POST /api/rag/huggingface/query` - Hugging Face
* `POST /api/rag/hf/query` - Hugging Face (alias)

### **Document Management**

* `POST /api/rag/documents/add` - Add document (JSON)
* `POST /api/rag/documents/add-file` - Upload file
* `POST /api/rag/documents/seed` - Seed default data
* `POST /api/rag/documents/update` - Update with versioning
* `GET /api/rag/documents/list` - List with filtering

### **Authentication**

* `POST /api/auth/token` - JWT login

---

# 🧩 **9. Project Goals**

This project is built for **AI engineers learning**:

* RAG pipelines
* RBAC security enforcement
* Local LLM inference
* Chat memory handling
* Metadata-aware document retrieval
* Enterprise knowledge systems
* End-to-end offline AI stack

It simulates what a **real company AI assistant** would look like.

---

# ⚙️ **10. What Makes This Project Unique**

* **Multi-provider architecture** - unified interface for local and cloud LLMs
* **Offline-first design** - works without internet using local models
* **Cloud integration** - seamless API integration when needed
* **Flexible RBAC** - level-based hierarchy with role overrides across all providers
* **Common vector space** - same embeddings and documents for all providers
* **Provider abstraction** - easy to add new LLM providers
* **Enterprise-grade** - flexible RBAC with level validation and role overrides
* **Session continuity** - conversation history works across providers
* **Prompt optimization** - token budgeting with smart context truncation
* **Debug capabilities** - complete prompt/response logging for optimization

---

# ✔️ **Summary**

This project is a **complete multi-provider enterprise RAG system**, combining:

* **Multiple LLM providers** (Local, Google, OpenAI, Hugging Face)
* **Unified RAG architecture** with shared document retrieval
* **Flexible RBAC** with level-based access and role overrides across all providers
* **Session memory** and conversation continuity
* **Document versioning** and metadata management
* **JWT authentication** with role-based access
* **FastAPI integration** with clean provider abstraction
* **Optimized prompt engineering** with token budgeting and debug exposure
* **Enhanced logging** with performance metrics and audit trails

### **Usage Examples**

```bash
# Local offline model
curl -X POST "/api/rag/local/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our company policy?", "use_llm": true}'

# Google Gemini
curl -X POST "/api/rag/google/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our company policy?", "use_llm": true}'

# OpenAI GPT  
curl -X POST "/api/rag/gpt/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our company policy?", "use_llm": true}'
```

Designed specifically for **learning AI engineering** and **multi-provider RAG architectures**.








