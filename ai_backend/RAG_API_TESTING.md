# RAG API Testing Guide

This document provides `curl` examples for testing the unified RAG API endpoints.
Ensure your FastAPI application is running before executing these commands.

**Base URL:** `http://localhost:5444` (or your application's host and port)

---

## 1. Querying RAG

### 1.1. Query Local RAG

**Endpoint:** `POST /api/rag/local/query`

```bash
curl -X POST "http://localhost:5444/api/rag/local/query" \
-H "Content-Type: application/json" \
-d 
'{ 
  "question": "What is the company mission?",
  "use_llm": true,
  "top_k": 3
}'
```

### 1.2. Query Google RAG

**Endpoint:** `POST /api/rag/google/query`
*(Requires `GOOGLE_API_KEY` to be set in your `.env` file)*

```bash
curl -X POST "http://localhost:5444/api/rag/google/query" \
-H "Content-Type: application/json" \
-d 
'{ 
  "question": "What is the company mission?",
  "use_llm": true,
  "top_k": 3
}'
```

---

## 2. Adding Documents

### 2.1. Add Document via JSON

**Endpoint:** `POST /api/rag/add`

```bash
curl -X POST "http://localhost:5444/api/rag/add" \
-H "Content-Type: application/json" \
-H "X-API-Key: test_admin_key" \
-d 
'{ 
  "source_name": "company_values.txt",
  "text": "Our core values are innovation, integrity, and customer satisfaction.",
  "metadata": {
    "department": "HR",
    "sensitivity": "public_internal",
    "allowed_roles": ["Employee", "Manager"]
  }
}'
```

### 2.2. Add Document via File Upload

**Endpoint:** `POST /api/rag/add-file`
*(Replace `/path/to/your/document.txt` with the actual path to a text file)*

```bash
curl -X POST "http://localhost:5444/api/rag/add-file" \
-H "X-API-Key: test_admin_key" \
-F "file=@/path/to/your/document.txt" \
-F "department=Engineering" \
-F "sensitivity=department_confidential"
```

---

## 3. Seeding and Clearing

### 3.1. Seed Default Documents

**Endpoint:** `POST /api/rag/seed`

```bash
# Seed without forcing re-ingestion if already populated
curl -X POST "http://localhost:5444/api/rag/seed" \
-H "X-API-Key: test_admin_key"

# Force re-ingestion of default documents (may create duplicates)
curl -X POST "http://localhost:5444/api/rag/seed?reseed=true" \
-H "X-API-Key: test_admin_key"
```

### 3.2. Clear RAG Collection

**Endpoint:** `POST /api/rag/clear`
*(Requires 'Executive' or 'Legal' role API key)*

```bash
curl -X POST "http://localhost:5444/api/rag/clear" \
-H "X-API-Key: test_executive_key"
```

---

## 4. Support Chat Session Management

### 4.1. Start a Support Session

**Endpoint:** `POST /api/rag/session/start`

```bash
curl -X POST "http://localhost:5444/api/rag/session/start" \
-H "X-API-Key: test_employee_key"
# Response will include a session_id. Use this in subsequent queries.
```

### 4.2. End a Support Session

**Endpoint:** `POST /api/rag/session/end`
*(Replace `YOUR_SESSION_ID` with the actual session ID obtained from starting a session)*

```bash
curl -X POST "http://localhost:5444/api/rag/session/end" \
-H "Content-Type: application/json" \
-H "X-API-Key: test_employee_key" \
-d 
'{ 
  "session_id": "YOUR_SESSION_ID"
}'
```

---

## 5. Sentiment Analysis

### 5.1. Analyze Sentiment

**Endpoint:** `POST /api/rag/sentiment`

```bash
curl -X POST "http://localhost:5444/api/rag/sentiment" \
-H "Content-Type: application/json" \
-d 
'{ 
  "text": "I am very happy with the service provided today."
}'
```

### 5.2. Get Sentiment Statistics

**Endpoint:** `GET /api/rag/sentiment/stats`

```bash
curl -X GET "http://localhost:5444/api/rag/sentiment/stats"
```
