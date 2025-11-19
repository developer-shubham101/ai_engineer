# RAG Chat React (Vite)

This is a small React app (Vite) that implements the RAG Chat UI split into components.

## Quick start

1. Install dependencies:

```bash
npm install
```

2. Run dev server:

```bash
npm run dev
```

3. Open the URL shown by Vite (usually http://localhost:3000).

## Notes
- Edit `src/components/RAGChat.jsx` to change `BASE_API_URL` if your backend is different (default: http://localhost:8000/api/local).
- Backend must accept `X-API-Key` header and implement endpoints: POST /query, POST /add, POST /add-file, POST /request-access, GET /access-requests, POST /update-metadata.
- If your backend is on a different origin, enable CORS or run a proxy.
