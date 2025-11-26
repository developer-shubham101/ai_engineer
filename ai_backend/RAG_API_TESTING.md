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

---

## 6. RBAC (Role-Based Access Control) Testing

### 6.1. Authentication

**Endpoint:** `POST /api/auth/token`

```bash
# Login as admin
curl -X POST "http://localhost:5444/api/auth/token" \
-H "Content-Type: application/json" \
-d '{
  "username": "admin",
  "password": "admin123"
}'

# Response includes access_token - use in subsequent requests
```

### 6.2. Add Document with Metadata Validation

**Endpoint:** `POST /api/rag/documents/add`

```bash
# ✅ Valid: Employee creating public_internal document
curl -X POST "http://localhost:5444/api/rag/documents/add" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_TOKEN" \
-d '{
  "source_name": "team_guidelines.md",
  "text": "Team collaboration guidelines...",
  "metadata": {
    "department": "Engineering",
    "sensitivity": "public_internal",
    "tags": "guidelines,team"
  }
}'

# ❌ Invalid: Employee trying to create highly_confidential
curl -X POST "http://localhost:5444/api/rag/documents/add" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer EMPLOYEE_TOKEN" \
-d '{
  "source_name": "confidential_doc.md",
  "text": "Confidential content...",
  "metadata": {
    "department": "Engineering",
    "sensitivity": "highly_confidential"
  }
}'
# Response: 403 - "Your role 'Employee' cannot create documents with sensitivity 'highly_confidential'"
```

### 6.3. Update Document with Department Ownership Check

**Endpoint:** `POST /api/rag/documents/update`

```bash
# ✅ Valid: HR updating their own department document
curl -X POST "http://localhost:5444/api/rag/documents/update" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer HR_TOKEN" \
-d '{
  "document_id": "doc_abc123",
  "text": "Updated HR policy content...",
  "version_notes": "Updated leave policy details",
  "status": "published"
}'

# ❌ Invalid: Manager trying to update HR department document
curl -X POST "http://localhost:5444/api/rag/documents/update" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer MANAGER_TOKEN" \
-d '{
  "document_id": "doc_abc123",
  "text": "Attempting to update HR doc...",
  "status": "published"
}'
# Response: 403 - "You cannot update documents from department 'HR'"
```

### 6.4. Query with RBAC Filtering

**Endpoint:** `POST /api/rag/local/query`

```bash
# Query as Employee - will only see public_internal documents
curl -X POST "http://localhost:5444/api/rag/local/query" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer EMPLOYEE_TOKEN" \
-d '{
  "question": "What is the HR leave policy?",
  "top_k": 5,
  "use_llm": false,
  "debug": true
}'

# Query as HR - will see department_confidential HR documents
curl -X POST "http://localhost:5444/api/rag/local/query" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer HR_TOKEN" \
-d '{
  "question": "What is the HR leave policy?",
  "top_k": 5,
  "use_llm": false,
  "debug": true
}'
```

### 6.5. Metadata Validation Examples

```bash
# ❌ Invalid department
curl -X POST "http://localhost:5444/api/rag/documents/add" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_TOKEN" \
-d '{
  "source_name": "test.md",
  "text": "Content...",
  "metadata": {
    "department": "InvalidDept",
    "sensitivity": "public_internal"
  }
}'
# Response: 400 - "Invalid department 'InvalidDept'"

# ❌ Invalid sensitivity level
curl -X POST "http://localhost:5444/api/rag/documents/add" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer YOUR_TOKEN" \
-d '{
  "source_name": "test.md",
  "text": "Content...",
  "metadata": {
    "department": "Engineering",
    "sensitivity": "super_secret"
  }
}'
# Response: 400 - "Invalid sensitivity 'super_secret'"

# ❌ Personal document without owner_id
curl -X POST "http://localhost:5444/api/rag/documents/add" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer HR_TOKEN" \
-d '{
  "source_name": "personal_record.md",
  "text": "Personal content...",
  "metadata": {
    "department": "HR",
    "sensitivity": "personal"
  }
}'
# Response: 400 - "Personal documents must have an 'owner_id' field"

# ✅ Valid personal document with owner_id
curl -X POST "http://localhost:5444/api/rag/documents/add" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer HR_TOKEN" \
-d '{
  "source_name": "personal_record.md",
  "text": "Personal content...",
  "metadata": {
    "department": "HR",
    "sensitivity": "personal",
    "owner_id": "u_emp_1"
  }
}'
```

### 6.6. Audit Log Examples

Check server logs for these audit entries:

- `RBAC_ACCESS_DENIED`: When users try to access restricted documents
- `RBAC_UPDATE_DENIED`: When users try to update cross-department documents
- `METADATA_VALIDATION_FAILED`: When invalid metadata is submitted
- `DOCUMENT_CREATED`: Successful document creation
- `DOCUMENT_UPDATED`: Successful document updates
- `METADATA_CHANGE`: When sensitivity levels are changed

---

## 7. Role-Based Sensitivity Permissions

| Role | Allowed Sensitivity Levels |
|------|---------------------------|
| Guest | `public_internal` |
| Employee | `public_internal` |
| Manager | `public_internal`, `department_confidential` |
| HR | `public_internal`, `department_confidential`, `role_confidential`, `personal` |
| SuperAdmin | ALL (including `highly_confidential`) |

## 8. Valid Metadata Fields

- **department**: `General`, `HR`, `Finance`, `Engineering`, `IT`, `Legal`, `Executive`, `Admin`
- **sensitivity**: `public_internal`, `department_confidential`, `role_confidential`, `highly_confidential`, `personal`
- **allowed_roles**: Array of valid roles: `["SuperAdmin", "HR", "Manager", "Employee", "Guest"]`
- **owner_id**: Required for `personal` sensitivity documents
- **tags**: Comma-separated string for searchability
- **public_summary**: Fallback text shown when full content is restricted

