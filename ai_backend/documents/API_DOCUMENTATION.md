# 🔌 API Documentation & Testing Guide

## Overview

This document provides comprehensive API documentation and testing examples for the Multi-Provider Enterprise RAG System. The system provides both REST API endpoints and interactive testing capabilities.

**Base URL:** `http://localhost:8000` (or your configured host and port)

## 📋 General Endpoints

### Health Check
- **GET** `/` - Health check endpoint
  - Returns: `{"status": "ok", "message": "Welcome to the AI Engineering API!"}`

## 🔐 Authentication & Authorization

### JWT Authentication
- **POST** `/api/auth/token` - Login and get JWT token

```bash
curl -X POST "http://localhost:5444/api/auth/token" \
-H "Content-Type: application/json" \
-d '{
  "username": "admin",
  "password": "admin123"
}'
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "user": {
    "user_id": "u_admin_1",
    "username": "admin",
    "role": "SuperAdmin",
    "department": "Executive",
    "profile": {
      "name": "Admin User",
      "email": "admin@company.com"
    }
  }
}
```

### API Key Authentication
- **Header**: `X-API-Key` or `Authorization: Bearer <token>`
- **Roles**: Employee, Manager, HR, Legal, Executive, Guest (no key)
- **Departments**: Engineering, Finance, HR, Legal, IT, Executive, General

### RBAC Sensitivity Levels
1. `public_internal` - All authenticated users
2. `department_confidential` - Same department or HR/Legal/Executive
3. `role_confidential` - Specific roles or HR/Legal/Executive
4. `highly_confidential` - Legal/Executive only
5. `personal` - Owner or HR/Legal/Executive

## 🤖 RAG Services (Primary API)

### Query & Retrieval

#### Main RAG Query Endpoint
- **POST** `/api/rag/{model_provider}/query` - Main RAG query endpoint
- **Path Parameter**: `model_provider` - `"local"`, `"google"`, `"gpt"`, or `"huggingface"`

**Request Body:**
```json
{
  "question": "What are the company leave policies?",
  "top_k": 3,
  "use_documents": true,
  "use_llm": true,
  "max_tokens": 256,
  "temperature": 0.1,
  "category": "optional_string",
  "debug": false,
  "local_llm_model": "phi2"
}
```

**Request Parameters:**
- `question` (string, required): The user's question or query
- `top_k` (integer, default: 3): Number of documents to retrieve
- `use_documents` (boolean, default: true): Whether to retrieve documents from the vector store. If false, the query will be sent directly to the LLM without RAG.
- `use_llm` (boolean, default: false): Whether to use LLM for response generation
- `max_tokens` (integer, default: 256): Maximum tokens for LLM response
- `temperature` (float, default: 0.1): Controls response creativity (0.0-1.0). Accepted for all LLM providers.
  - `0.0`: Deterministic, factual responses
  - `0.1`: Default balanced responses
  - `0.5`: Moderate creativity
  - `1.0`: Maximum creativity
- `category` (string, optional): Category filter for documents
- `debug` (boolean, default: false): Include debug information in response
- `local_llm_model` (string, optional): Specific model for local provider

**Features:**
- RBAC filtering based on user role/department
- Session-aware with conversation history
- Onboarding flow support
- Tone-aware responses
- Multi-turn chat support

#### Local Model Query (Authenticated)
```bash
curl -X POST "http://localhost:5444/api/rag/local/query" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "question": "What are the company leave policies?",
  "top_k": 3,
  "use_documents": true,
  "use_llm": true,
  "temperature": 0.1
}'
```

#### Guest/Unauthenticated Query
```bash
curl -X POST "http://localhost:5444/api/rag/local/query" \
-H "Content-Type: application/json" \
-d '{
  "question": "What are your office hours?",
  "top_k": 3,
  "use_llm": true,
  "temperature": 0.0
}'
```

#### Google Model Query
```bash
curl -X POST "http://localhost:5444/api/rag/google/query" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "question": "What is the company mission?",
  "use_llm": true,
  "top_k": 3,
  "temperature": 0.3
}'
```

**Response Format:**
```json
{
  "answer": "Annual leave is 20 days per year...",
  "retrieved": [
    {
      "id": "doc_123",
      "text": "Leave policy document...",
      "metadata": {...},
      "distance": 0.85
    }
  ],
  "context": "Combined context from retrieved documents",
  "final_prompt": "System: You are an HR assistant..." // Debug mode
}
```

## 📄 Document Management

### Add Document via JSON
- **POST** `/api/rag/documents/add`

```bash
curl -X POST "http://localhost:5444/api/rag/documents/add" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "source_name": "company_values.txt",
  "text": "Our core values are innovation, integrity, and customer satisfaction.",
  "metadata": {
    "department": "HR",
    "sensitivity": "public_internal",
    "allowed_roles": ["Employee", "Manager"]
  }
}'
```

### Add Document via File Upload
- **POST** `/api/rag/documents/add-file`
- **Supported Formats**: `.md`, `.markdown`, `.html`, `.htm`, `.json`, `.txt`
- **Max Size**: 5MB

```bash
curl -X POST "http://localhost:5444/api/rag/documents/add-file" \
-H "Authorization: Bearer $TOKEN" \
-F "file=@/path/to/document.txt" \
-F "department=Engineering" \
-F "sensitivity=department_confidential"
```

### Document Versioning

#### Update Document (Creates New Version)
```bash
curl -X POST "http://localhost:5444/api/rag/documents/update" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "document_id": "doc_abc123...",
  "text": "Company Leave Policy v2.0\n\nEmployees get 20 days annual leave (updated!).",
  "version_notes": "Increased annual leave from 15 to 20 days",
  "status": "published"
}'
```

#### List All Documents
```bash
curl -X GET "http://localhost:5444/api/rag/documents/list?latest_only=true" \
-H "Authorization: Bearer $TOKEN"
```

#### Get Version History
```bash
curl -X GET "http://localhost:5444/api/rag/documents/doc_abc123.../versions" \
-H "Authorization: Bearer $TOKEN"
```

#### Compare Two Versions
```bash
curl -X GET "http://localhost:5444/api/rag/documents/doc_abc123.../compare?version1=1.0&version2=2.0" \
-H "Authorization: Bearer $TOKEN"
```

### Seed and Clear Operations

#### Seed Default Documents
```bash
# Seed without forcing re-ingestion
curl -X POST "http://localhost:5444/api/rag/documents/seed" \
-H "Authorization: Bearer $TOKEN"

# Force re-ingestion
curl -X POST "http://localhost:5444/api/rag/documents/seed?reseed=true" \
-H "Authorization: Bearer $TOKEN"
```

#### Clear RAG Collection
```bash
curl -X POST "http://localhost:5444/api/rag/documents/clear" \
-H "Authorization: Bearer $TOKEN"
```

## 🎭 Sentiment Analysis

### Analyze Text Sentiment and Tone
- **POST** `/api/rag/sentiment`

```bash
curl -X POST "http://localhost:5444/api/rag/sentiment" \
-H "Content-Type: application/json" \
-d '{
  "text": "I am very happy with the service provided today."
}'
```

**Response:**
```json
{
  "ok": true,
  "result": {
    "text": "I am very happy with the service provided today.",
    "sentiment": "positive",
    "tone": "polite",
    "proba": {
      "sentiment": {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
      "tone": {"polite": 0.7, "neutral": 0.3}
    }
  }
}
```

### Get Sentiment Statistics
```bash
curl -X GET "http://localhost:5444/api/rag/sentiment/stats"
```

## 🧪 RBAC Testing Examples

### Valid Operations

#### Employee Creating Public Document
```bash
curl -X POST "http://localhost:5444/api/rag/documents/add" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer EMPLOYEE_TOKEN" \
-d '{
  "source_name": "team_guidelines.md",
  "text": "Team collaboration guidelines...",
  "metadata": {
    "department": "Engineering",
    "sensitivity": "public_internal",
    "tags": "guidelines,team"
  }
}'
```

#### HR Updating Department Document
```bash
curl -X POST "http://localhost:5444/api/rag/documents/update" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer HR_TOKEN" \
-d '{
  "document_id": "doc_abc123",
  "text": "Updated HR policy content...",
  "version_notes": "Updated leave policy details",
  "status": "published"
}'
```

### Invalid Operations (Will Return 403)

#### Employee Trying to Create Highly Confidential Document
```bash
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

#### Manager Trying to Update HR Document
```bash
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

## 📊 Metadata Validation

### Valid Metadata Fields
- **department**: `General`, `HR`, `Finance`, `Engineering`, `IT`, `Legal`, `Executive`, `Admin`
- **sensitivity**: `public_internal`, `department_confidential`, `role_confidential`, `highly_confidential`, `personal`
- **allowed_roles**: Array of valid roles
- **owner_id**: Required for `personal` sensitivity documents
- **tags**: Comma-separated string for searchability
- **public_summary**: Fallback text shown when full content is restricted

### Validation Examples

#### Invalid Department
```bash
curl -X POST "http://localhost:5444/api/rag/documents/add" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "source_name": "test.md",
  "text": "Content...",
  "metadata": {
    "department": "InvalidDept",
    "sensitivity": "public_internal"
  }
}'
# Response: 400 - "Invalid department 'InvalidDept'"
```

#### Personal Document Without Owner ID
```bash
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
```

## 🌡️ Temperature Control Examples

### Deterministic Response (Temperature = 0.0)
```bash
curl -X POST "http://localhost:5444/api/rag/local/query" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "question": "What is our exact leave policy?",
  "use_llm": true,
  "temperature": 0.0
}'
```

### Balanced Response (Temperature = 0.1 - Default)
```bash
curl -X POST "http://localhost:5444/api/rag/google/query" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "question": "Explain our company benefits",
  "use_llm": true,
  "temperature": 0.1
}'
```

### Creative Response (Temperature = 0.7)
```bash
curl -X POST "http://localhost:5444/api/rag/gpt/query" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "question": "Write a creative summary of our mission",
  "use_llm": true,
  "temperature": 0.7,
  "max_tokens": 512
}'
```

### Temperature Guidelines by Use Case

| Temperature | Use Case | Example |
|-------------|----------|----------|
| 0.0 | Compliance, Legal | "What is the exact policy?" |
| 0.1 | General Q&A | "How do I request leave?" |
| 0.3 | Explanations | "Explain our benefits package" |
| 0.5 | Summaries | "Summarize the quarterly report" |
| 0.7 | Creative Content | "Write a team announcement" |
| 1.0 | Brainstorming | "Generate ideas for team building" |

## 🔍 Debug and Monitoring

### Query with Debug Information
```bash
curl -X POST "http://localhost:5444/api/rag/local/query" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "question": "What is the HR leave policy?",
  "top_k": 5,
  "use_llm": true,
  "temperature": 0.2,
  "debug": true
}'
```

**Debug Response includes:**
```json
{
  "answer": "Generated response...",
  "retrieved": [...],
  "context": "Combined context...",
  "final_prompt": "System: Assistant for Saarthi Infotech | SuperAdmin/Executive | Admin User | Prev: What is...\n\nContext: Leave policy documents...\n\nQuestion: What is the HR leave policy?"
}
```

### Audit Log Events
Check server logs for these audit entries:
- `RBAC_ACCESS_DENIED`: When users try to access restricted documents
- `RBAC_UPDATE_DENIED`: When users try to update cross-department documents
- `METADATA_VALIDATION_FAILED`: When invalid metadata is submitted
- `DOCUMENT_CREATED`: Successful document creation
- `DOCUMENT_UPDATED`: Successful document updates
- `METADATA_CHANGE`: When sensitivity levels are changed

## 🚀 Legacy Endpoints (Direct LLM Services)

### Text Generation
```bash
# Summarize text
curl -X POST "http://localhost:5444/summarize" \
-H "Content-Type: application/json" \
-d '{"text": "Long text content here..."}'

# Generate text
curl -X POST "http://localhost:5444/generate" \
-H "Content-Type: application/json" \
-d '{"text": "Write a haiku about coding."}'

# Conversational chat
curl -X POST "http://localhost:5444/chat" \
-H "Content-Type: application/json" \
-d '{"user_input": "Hello, who are you?"}'
```

### External LLM Services
```bash
# OpenAI generation
curl -X POST "http://localhost:5444/generate/openai" \
-H "Content-Type: application/json" \
-d '{"text": "Explain quantum computing"}'

# Hugging Face generation
curl -X POST "http://localhost:5444/generate/hf" \
-H "Content-Type: application/json" \
-d '{"text": "Generate a story about AI"}'
```

## 📝 Role-Based Sensitivity Permissions

| Role | Allowed Sensitivity Levels |
|------|------------------------------|
| Guest | `public_internal` |
| Employee | `public_internal` |
| Employee L1 | `public_internal` |
| Employee L2 | `public_internal` |
| Manager | `public_internal`, `department_confidential` |
| HR | `public_internal`, `department_confidential`, `role_confidential`, `personal` |
| SuperAdmin | ALL (including `highly_confidential`) |

## 🔧 Environment Setup

### Set Token as Environment Variable
```bash
# Linux/Mac
export TOKEN="your_jwt_token_here"

# Windows PowerShell
$env:TOKEN="your_jwt_token_here"

# Windows CMD
set TOKEN=your_jwt_token_here
```

### Quick Test Script
```bash
#!/bin/bash
# Test basic functionality

# 1. Get token
TOKEN=$(curl -s -X POST "http://localhost:5444/api/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | \
  jq -r '.access_token')

# 2. Test query with temperature
curl -X POST "http://localhost:5444/api/rag/local/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"question": "What is the company mission?", "use_llm": true, "temperature": 0.1}'

# 3. Test document add
curl -X POST "http://localhost:5444/api/rag/documents/add" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{
    "source_name": "test_doc.md",
    "text": "This is a test document for API validation.",
    "metadata": {
      "department": "General",
      "sensitivity": "public_internal"
    }
  }'
```

## 🧪 Temperature Testing Script

```bash
#!/bin/bash
# Test temperature parameter across all providers

# Get authentication token
TOKEN=$(curl -s -X POST "http://localhost:5444/api/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | \
  jq -r '.access_token')

echo "Testing temperature parameters across providers..."

# Test different temperatures
for temp in 0.0 0.1 0.5 1.0; do
  echo "\n=== Testing Temperature: $temp ==="
  
  # Local provider
  echo "Local Provider:"
  curl -s -X POST "http://localhost:5444/api/rag/local/query" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"question\": \"What is our policy?\", \"use_llm\": false, \"temperature\": $temp}" | \
    jq -r '.answer // "Success"'
  
  # Google provider (if API key available)
  echo "Google Provider:"
  curl -s -X POST "http://localhost:5444/api/rag/google/query" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer $TOKEN" \
    -d "{\"question\": \"What is our policy?\", \"use_llm\": false, \"temperature\": $temp}" | \
    jq -r '.answer // "Success"'
done

echo "\nTemperature testing complete!"
```

This API documentation provides comprehensive coverage of all available endpoints, authentication methods, temperature control, and testing scenarios for the Multi-Provider Enterprise RAG System.