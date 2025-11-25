# COPILOT_CONTEXT.md

## 1. Project Summary
- A lightweight React frontend for testing and interacting with a role-based enterprise RAG (Retrieval-Augmented Generation) backend.
- Provides a chat interface, file upload and JSON document ingestion UIs, admin panels for access-requests and metadata updates, and session persistence.
- Sends every request with `X-API-Key` and exposes a UI role selector to simulate role-based behaviors (Employee, Manager, HR, Legal, Executive).
- Designed primarily as a developer tool to exercise backend endpoints and demonstrate RAG behavior (retrieved docs, filtered results, public summaries).

## 2. High-Level Architecture
- `index.html` + Vite bootstrapped React app (development server: `vite`).
- `src/`
  - `main.jsx` — React entry.
  - `App.jsx` — top-level container.
  - `components/` — reusable components (RAGChat, AddJsonForm, UploadFileForm, UpdateMetadataForm, ToastList).
  - `styles.css` — small project-specific styles.
  - `utility/const.js` - Defines constants like `BASE_API_URL`.
- Interacts with backend via REST endpoints at configurable `BASE_API_URL` (default `http://localhost:8000`).
  - API paths now include `/api/rag/{model_provider}/` (e.g., `/api/rag/local/query`).
  - `x-session-id` is included in request headers for session management.
- Backend responsibilities (outside repo scope but assumed):
  - Auth: validate `X-API-Key`, map to roles.
  - RAG: query vector DB (Chroma/FAISS/Pinecone) and optionally a LLM.
  - Document ingestion: accept `/add`, `/add-file`.
  - Access workflow: `/request-access`, `/access-requests`, `/update-metadata`.
  - Session management: `/api/rag/session/start` and `/api/rag/session/end`.
- Libraries & tools:
  - React 18, Vite, Bootstrap 5 (CDN), Tailwind (optional utilities CDN).
  - Fetch API for network requests (no jQuery in React version).
- Data stores assumed:
  - Primary DB for metadata and access requests (Postgres/SQLite).
  - Vector DB for embeddings (Chroma/FAISS/Pinecone/Milvus).
  - Optional LLM service (local or hosted).

## 3. Key APIs / Functions
> These are the primary endpoints/features the frontend expects from the backend.

- `POST /api/rag/{model_provider}/query` — `{ question, top_k, use_llm }` → `{ answer, retrieved[], context, filtered_out_count?, public_summaries?, filtered_details? }`
- `POST /api/rag/{model_provider}/add` — `{ source_name, text, metadata }` → add JSON doc, returns status / ids
- `POST /api/rag/{model_provider}/add-file` — multipart/form-data (file, department, sensitivity, tags, public_summary, owner_id) → upload + chunking result
- `POST /request-access` — `{ document_id?, source_name?, reason? }` → submit access request
- `GET /access-requests` — admin list of pending access requests
- `POST /update-metadata` — `{ ids: [...], metadata: {...} }` → update chunk metadata
- `POST /api/rag/session/start` — Start a new session, returns `x-session-id`.
- `POST /api/rag/session/end` — End an existing session.
- `RAGChat.sendQuery(question)` — client-side wrapper for network call to `/api/rag/{model_provider}/query`
- `RAGChat.postAddJson(payload)` — client-side wrapper for `/api/rag/{model_provider}/add`
- `RAGChat.postUploadFile(formData)` — client-side wrapper for `/api/rag/{model_provider}/add-file`
- `RAGChat.requestAccess(payload)` — client-side wrapper for `/request-access`
- `RAGChat.startSession()` — client-side wrapper for `/api/rag/session/start`
- `RAGChat.endSession()` — client-side wrapper for `/api/rag/session/end`

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
- `src/App.jsx` — top-level app container (renders `RAGChat`).
- `src/components/RAGChat.jsx` — core chat UI, session state, and main network wrappers. Includes session management and model provider selection.
- `src/components/AddJsonForm.jsx` — modal form to POST `/add`.
- `src/components/UploadFileForm.jsx` — modal form to post `/add-file` (multipart).
- `src/components/UpdateMetadataForm.jsx` — modal to update chunk metadata.
- `src/components/ToastList.jsx` — transient UI toasts for errors / notifications.
- `src/styles.css` — small custom styles used across components.
- `src/utility/const.js` — Defines constants such as `BASE_API_URL`.
- `README.md` — run/build instructions; dev notes and backend expectations.

## 6. Data Models / Structures
- **Query request**
```json
{
  "question": "string",
  "top_k": 3,
  "use_llm": false
}