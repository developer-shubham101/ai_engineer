# COPILOT_CONTEXT.md

## 1. Project Summary
- A lightweight React frontend for testing and interacting with a role-based enterprise RAG (Retrieval-Augmented Generation) backend.
- Provides a chat interface with **persistent conversation history**, file upload and JSON document ingestion UIs, admin panels for access-requests and metadata updates.
- Uses JWT token-based authentication with username/password login. All requests include `Authorization: Bearer <token>` header.
- User roles are determined by the backend based on credentials and embedded in the JWT token (Employee, Manager, HR, Legal, Executive, SuperAdmin).
- **ChatGPT-like conversation management**: Users can create, switch between, rename, and delete conversations. All messages are persisted to the backend with full RAG pipeline logging.
- Designed primarily as a developer tool to exercise backend endpoints and demonstrate RAG behavior (retrieved docs, filtered results, public summaries).

## 2. High-Level Architecture
- `index.html` + Vite bootstrapped React app (development server: `vite`).
- `src/`
  - `main.jsx` — React entry.
  - `App.jsx` — top-level container with authentication routing.
  - `components/` — reusable components (Login, RAGChat, ConversationSidebar, ConversationMessageDetail, PromptTemplateManager, AddJsonForm, UploadFileForm, UpdateMetadataForm, ToastList).
  - `styles.css` — project styles including conversation sidebar, message details, and responsive layout.
  - `utility/const.js` - Defines constants like `BASE_API_URL`.
  - `utility/auth.js` - JWT token management utilities (encode/decode, localStorage operations).
- Interacts with backend via REST endpoints at configurable `BASE_API_URL` (default `http://localhost:8000`).
  - API paths include `/api/rag/{model_provider}/query` for queries and `/api/rag/documents/*` for document management.
  - `X-Session-ID` is extracted from JWT token and included in request headers automatically.
  - Local LLM model selection available for local provider queries.
- Backend responsibilities (outside repo scope but assumed):
  - Auth: JWT token generation via `/api/auth/token`, validate Bearer tokens, embed user info and session_id in token.
  - RAG: query vector DB (ChromaDB) with multi-provider LLM support (Local/Google/OpenAI/HuggingFace).
  - Document management: comprehensive CRUD with versioning, RBAC filtering, and metadata validation.
  - Access workflow: `/request-access`, `/access-requests`, role-based document filtering.
  - Session management: automatic via JWT token (session_id embedded in token payload).
  - Conversation management: `/api/conversations/*` endpoints for CRUD operations on conversations and messages.
- Libraries & tools:
  - React 18, Vite, Bootstrap 5 (CDN), Tailwind (optional utilities CDN).
  - Fetch API for network requests (no jQuery in React version).
- Data stores assumed:
  - Primary DB for metadata and access requests (Postgres/SQLite).
  - Vector DB for embeddings (Chroma/FAISS/Pinecone/Milvus).
  - Optional LLM service (local or hosted).

## 3. Key APIs / Functions
> These are the primary endpoints/features the frontend expects from the backend.

**Authentication:**
- `POST /api/auth/token` — `{ username, password }` → `{ access_token, token_type, user: { user_id, username, role, department, profile } }`

**Prompt Templates (require Bearer token):**
- `GET /api/templates` — list all templates
- `POST /api/templates` — create new template
- `PUT /api/templates/{name}` — update template
- `DELETE /api/templates/{name}` — delete template

**Conversation History (NEW - require Bearer token):**
- `GET /api/conversations` — list all conversations for authenticated user
- `POST /api/conversations` — create new conversation with optional title
- `GET /api/conversations/{conversation_id}` — get conversation details
- `PUT /api/conversations/{conversation_id}` — update conversation (rename)
- `DELETE /api/conversations/{conversation_id}` — delete conversation (soft delete)
- `GET /api/conversations/{conversation_id}/messages` — get all messages with full RAG logging
- `POST /api/conversations/{conversation_id}/restore` — restore conversation to current session

**RAG Operations (require Bearer token):**
- `POST /api/rag/{model_provider}/query` — `{ question, top_k, use_llm, max_tokens?, category?, local_llm_model?, conversation_id? }` → `{ answer, retrieved[], context, filtered_out_count?, public_summaries?, filtered_details? }`
  - `model_provider`: `local`, `google`, `gpt`, `hf` (huggingface)
  - `local_llm_model`: Optional parameter for local provider (e.g., "llama32-1b", "llama32-3b", "llama31-8b", "phi3-mini", "gemma2-2b", "distilgpt2-company-tuned")
  - `conversation_id`: Optional parameter to associate query with a conversation

**Document Management (require Bearer token):**
- `POST /api/rag/documents/add` — `{ source_name, text, metadata }` → add JSON doc with versioning
- `POST /api/rag/documents/add-file` — multipart/form-data (file, department, sensitivity, tags, public_summary, owner_id) → upload + chunking with versioning
- `POST /api/rag/documents/update` — `{ document_id, text, version_notes?, status? }` → create new version (non-destructive)
- `POST /api/rag/documents/seed` — seed documents from data folder
- `GET /api/rag/documents/list` — list documents with filtering (department, status, latest_only)
- `GET /api/rag/documents/{document_id}/versions` — get version history
- `GET /api/rag/documents/{document_id}/versions/{version}` — get specific version
- `GET /api/rag/documents/{document_id}/compare?version1=1.0&version2=2.0` — compare versions
- `POST /api/rag/documents/{document_id}/archive` — archive version
- `POST /request-access` — `{ document_id?, source_name?, reason? }` → submit access request
- `GET /access-requests` — admin list of pending access requests
- `POST /update-metadata` — `{ ids: [...], metadata: {...} }` → update chunk metadata

**Client-side functions:**
- `Login.handleLogin(username, password)` — authenticate and store JWT token
- `RAGChat.sendQuery(question)` — client-side wrapper for network call to `/api/rag/{model_provider}/query` (includes local_llm_model and conversation_id)
- `RAGChat.loadConversations()` — fetch all conversations for current user
- `RAGChat.createNewConversation(title)` — create new conversation
- `RAGChat.loadConversationMessages(conversationId)` — load messages from specific conversation
- `RAGChat.switchConversation(conversationId)` — switch to different conversation
- `RAGChat.renameConversation(conversationId, newTitle)` — rename conversation
- `RAGChat.deleteConversation(conversationId)` — delete conversation
- `RAGChat.postAddJson(payload)` — client-side wrapper for `/api/rag/documents/add`
- `RAGChat.postUploadFile(formData)` — client-side wrapper for `/api/rag/documents/add-file`
- `RAGChat.listDocuments(filters)` — client-side wrapper for `/api/rag/documents/list`
- `RAGChat.getDocumentVersions(documentId)` — client-side wrapper for version history
- `RAGChat.updateDocument(payload)` — client-side wrapper for `/api/rag/documents/update`
- `RAGChat.seedDocuments()` — client-side wrapper for `/api/rag/documents/seed`
- `RAGChat.compareDocumentVersions(documentId, version1, version2)` — client-side wrapper for version comparison
- `RAGChat.archiveDocumentVersion(documentId, version)` — client-side wrapper for version archiving
- `RAGChat.testRBACAccess()` — test RBAC restrictions by attempting to create highly confidential document
- `RAGChat.requestAccess(payload)` — client-side wrapper for `/request-access`
- `RAGChat.handleLogout()` — clear authentication, conversations, and return to login

## 4. Coding Conventions
- Languages & versions:
  - JavaScript / React (ESM). Node >= 18 recommended for dev server.
  - Vite for bundling / dev server.
- Type & linting:
  - Project is JavaScript (not TypeScript) by default; prefer explicit runtime checks for JSON parsing.
  - If migrating to TS, follow strict mode with `noImplicitAny`.
- Formatting:
  - Use Prettier / ESLint recommended defaults (2 spaces indentation, semicolons optional but consistent).
  - Keep components small and functional; prefer hooks (`useState`, `useEffect`, `useRef`).
- Function & component style:
  - Functional components with named exports (default export for main component).
  - Keep side effects in `useEffect`.
  - Handler function names: `handleXxx` for UI events, `fetchXxx` / `postXxx` for network calls.
  - Keep network layer in components or dedicated hooks (thin wrappers).

## 5. Important Files & Their Purpose
- `index.html` — HTML entry; includes Bootstrap & Tailwind CDNs and injects `#root`.
- `vite.config.js` / `vite.config.mjs` — Vite configuration (ensure ESM compatibility or set `"type":"module"`).
- `package.json` — dependencies and NPM scripts (`dev`, `build`, `preview`).
- `src/main.jsx` — application bootstrap, mounts React tree.
- `src/App.jsx` — top-level app container with authentication routing (shows Login or RAGChat based on auth state).
- `src/components/Login.jsx` — login form with username/password fields and "Login with Guest" button (auto-fills guest/guest123), calls `/api/auth/token` and stores JWT.
- `src/components/RAGChat.jsx` — core chat UI with conversation history integration. Uses Bearer token authentication. Manages conversations, messages, and sidebar state. Includes local LLM model selection for local provider.
- `src/components/ConversationSidebar.jsx` — collapsible sidebar showing conversation list with create, rename, and delete functionality.
- `src/components/ConversationMessageDetail.jsx` — expandable component showing RAG pipeline details (retrieved docs, LLM config, embeddings, sentiment/tone).
- `src/components/PromptTemplateManager.jsx` — admin component to list, create, edit, and delete dynamic prompt templates.
- `src/components/AddJsonForm.jsx` — enhanced modal form to POST `/api/rag/documents/add` with RBAC metadata fields.
- `src/components/UploadFileForm.jsx` — modal form to POST `/api/rag/documents/add-file` (multipart).
- `src/components/UpdateMetadataForm.jsx` — modal to update chunk metadata.
- `src/components/DocumentVersionModal.jsx` — modal for document version management (history, comparison, archiving).
- `src/components/PersonalizedTestModal.jsx` — modal for testing personalized AI responses with different scenarios.
- `src/components/ToastList.jsx` — transient UI toasts for errors / notifications.
- `src/styles.css` — small custom styles used across components.
- `src/utility/const.js` — Defines constants such as `BASE_API_URL`.
- `src/utility/auth.js` — JWT token utilities (decode, expiration check, localStorage management).
- `README.md` — run/build instructions; dev notes and backend expectations.

## 6. Enhanced UI Components

**ConversationSidebar Features:**
- List all conversations with title, date, and message count
- Create new conversations with custom titles
- Rename conversations inline
- Delete conversations with confirmation
- Active conversation highlighting
- Responsive design (collapsible on mobile)
- Relative time formatting ("2h ago", "3d ago")

**ConversationMessageDetail Features:**
- Expandable RAG pipeline logging for assistant messages
- Retrieved documents with metadata and distance scores
- LLM configuration (provider, model, temperature, tokens)
- Embeddings information (model, dimensions)
- Retrieval settings (top_k, use_documents, use_llm)
- Full LLM prompt and raw response display
- Sentiment and tone badges with color coding
- Processing time metrics
- Error message display

**PromptTemplateManager Features:**
- List all prompt templates
- Create new templates with name and content
- Update existing templates
- Delete templates
- Variables support: `{source_docs}`, `{user_question}`, `{chat_history}`

**DocumentVersionModal Features:**
- Get version history for any document ID
- Compare two versions with unified diff
- Archive specific versions
- Console logging for detailed results

**PersonalizedTestModal Scenarios:**
- Guest Job Inquiry: Tests job matching for external users
- Career Guidance: Tests internal employee support
- HR Recruitment: Tests role-specific assistance
- Profile Analysis: Tests skill-based job matching
- Onboarding Flow: Tests guest user profile collection

**Enhanced AddJsonForm:**
- Simple mode: Dropdown fields for department, sensitivity, roles
- Advanced mode: Raw JSON metadata editing
- RBAC validation: Real-time field validation
- Personal document support: Owner ID field for personal docs
- Public summary field for restricted documents

**Local LLM Model Selection:**
- Dropdown appears when "Local" provider is selected
- Supports multiple local models: Llama 3.2 1B/3B, Llama 3.1 8B, Phi-3 Mini, Gemma 2 2B
- Model selection persisted in localStorage
- Automatically included in query payload for local providerblic summary: Fallback content for restricted access

## 7. Data Models / Structures

**Login request:**
```json
{
  "username": "guest",
  "password": "guest123"
}
```

**Login response:**
```json
{
  "access_token": "eyJhbGc...",
  "token_type": "bearer",
  "user": {
    "user_id": "u_admin_1",
    "username": "admin",
    "role": "SuperAdmin",
    "department": "Executive",
    "profile": {
      "gender": "Other",
      "location": "HQ",
      "name": "Admin User"
    }
  }
}
```

**JWT Token Payload (decoded):**
```json
{
  "user_id": "u_admin_1",
  "username": "admin",
  "role": "SuperAdmin",
  "department": "Executive",
  "session_id": "sess_d41a5bf69cd2474a84bf9e7853e27678",
  "exp": 1764178920,
  "iat": 1764092520
}
```

**Query request:**
```json
{
  "question": "string",
  "top_k": 3,
  "use_llm": false,
  "use_documents": true
}
```

**Request Headers (authenticated):**
```json
{
  "Content-Type": "application/json",
  "Authorization": "Bearer eyJhbGc...",
  "X-Session-ID": "sess_d41a5bf69cd2474a84bf9e7853e27678"
}
```

**Document Metadata:**
```json
{
  "source": "string",
  "department": "string",
  "sensitivity": "string",
  "allowed_roles": ["string"],
  "owner_id": "string",
  "public_summary": "string",
  "document_id": "string",
  "version": "string",
  "version_created_at": "string",
  "version_created_by": "string",
  "parent_version": "string",
  "status": "string",
  "is_latest_version": true
}
```

**Sensitivity Levels (RBAC):**
- `public_internal` - All authenticated users
- `department_confidential` - Same department or HR/Legal/Executive
- `role_confidential` - Specific roles or HR/Legal/Executive
- `highly_confidential` - Legal/Executive only
- `personal` - Owner or HR/Legal/Executive

**Document Status:**
- `draft` - Work in progress
- `pending_approval` - Awaiting review
- `published` - Active and searchable
- `archived` - Soft deleted

**Model Providers:**
- `local` - Local Mistral-7B model
- `google` - Google Gemini API
- `gpt` - OpenAI GPT API
- `hf` - Hugging Face Inference API

**New UI Features:**

**Use Docs toggle:**
- A checkbox to enable or disable document retrieval for RAG queries.

**Enhanced Document Management:**
- Document version history viewing
- Version comparison with diff display
- Version archiving functionality
- Advanced metadata editor with RBAC fields
- Document listing with filtering
- Seed documents from backend data folder

**RBAC Testing:**
- Test role-based access control restrictions
- Validate metadata requirements
- Test sensitivity level permissions
- Department ownership validation

**Personalized AI Testing:**
- Guest user job inquiry scenarios
- Internal employee career guidance
- HR manager recruitment assistance
- Profile analysis and job matching
- Guest onboarding flow testing

**Enhanced Authentication:**
- Chat history cleared on logout
- Session management with JWT tokens
- Multi-role user support

**Admin Panel Features:**
- List all documents with metadata
- Seed documents from backend
- Access request management
- RBAC permission testing
- Version management tools
- Personalized response testing