# COPILOT_CONTEXT.md

## 1. Project Summary
- A lightweight React frontend for testing and interacting with a role-based enterprise RAG (Retrieval-Augmented Generation) backend.
- Provides a chat interface, file upload and JSON document ingestion UIs, admin panels for access-requests and metadata updates, and session persistence.
- Uses JWT token-based authentication with username/password login. All requests include `Authorization: Bearer <token>` header.
- User roles are determined by the backend based on credentials and embedded in the JWT token (Employee, Manager, HR, Legal, Executive, SuperAdmin).
- Designed primarily as a developer tool to exercise backend endpoints and demonstrate RAG behavior (retrieved docs, filtered results, public summaries).

## 2. High-Level Architecture
- `index.html` + Vite bootstrapped React app (development server: `vite`).
- `src/`
  - `main.jsx` — React entry.
  - `App.jsx` — top-level container with authentication routing.
  - `components/` — reusable components (Login, RAGChat, AddJsonForm, UploadFileForm, UpdateMetadataForm, ToastList).
  - `styles.css` — small project-specific styles.
  - `utility/const.js` - Defines constants like `BASE_API_URL`.
  - `utility/auth.js` - JWT token management utilities (encode/decode, localStorage operations).
- Interacts with backend via REST endpoints at configurable `BASE_API_URL` (default `http://192.168.1.2:8000`).
  - API paths now include `/api/rag/{model_provider}/` (e.g., `/api/rag/local/query`).
  - `X-Session-ID` is extracted from JWT token and included in request headers automatically.
- Backend responsibilities (outside repo scope but assumed):
  - Auth: JWT token generation via `/api/auth/token`, validate Bearer tokens, embed user info and session_id in token.
  - RAG: query vector DB (Chroma/FAISS/Pinecone) and optionally a LLM.
  - Document ingestion: accept `/add`, `/add-file`.
  - Access workflow: `/request-access`, `/access-requests`, `/update-metadata`.
  - Session management: automatic via JWT token (session_id embedded in token payload).
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

**RAG Operations (require Bearer token):**
- `POST /api/rag/{model_provider}/query` — `{ question, top_k, use_llm }` → `{ answer, retrieved[], context, filtered_out_count?, public_summaries?, filtered_details? }`
- `POST /api/rag/{model_provider}/add` — `{ source_name, text, metadata }` → add JSON doc, returns status / ids
- `POST /api/rag/{model_provider}/add-file` — multipart/form-data (file, department, sensitivity, tags, public_summary, owner_id) → upload + chunking result
- `POST /request-access` — `{ document_id?, source_name?, reason? }` → submit access request
- `GET /access-requests` — admin list of pending access requests
- `POST /update-metadata` — `{ ids: [...], metadata: {...} }` → update chunk metadata

**Client-side functions:**
- `Login.handleLogin(username, password)` — authenticate and store JWT token
- `RAGChat.sendQuery(question)` — client-side wrapper for network call to `/api/rag/{model_provider}/query`
- `RAGChat.postAddJson(payload)` — client-side wrapper for `/api/rag/{model_provider}/add`
- `RAGChat.postUploadFile(formData)` — client-side wrapper for `/api/rag/{model_provider}/add-file`
- `RAGChat.requestAccess(payload)` — client-side wrapper for `/request-access`
- `RAGChat.handleLogout()` — clear authentication and return to login

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
- `src/components/RAGChat.jsx` — core chat UI and main network wrappers. Uses Bearer token authentication. Displays user info and logout button.
- `src/components/AddJsonForm.jsx` — modal form to POST `/add`.
- `src/components/UploadFileForm.jsx` — modal form to post `/add-file` (multipart).
- `src/components/UpdateMetadataForm.jsx` — modal to update chunk metadata.
- `src/components/ToastList.jsx` — transient UI toasts for errors / notifications.
- `src/styles.css` — small custom styles used across components.
- `src/utility/const.js` — Defines constants such as `BASE_API_URL`.
- `src/utility/auth.js` — JWT token utilities (decode, expiration check, localStorage management).
- `README.md` — run/build instructions; dev notes and backend expectations.

## 6. Data Models / Structures

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
  "use_llm": false
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