# REST API Endpoints - Current Implementation

## General Endpoints

### Health Check
- **GET** `/` - Health check endpoint
  - Returns: `{"status": "ok", "message": "Welcome to the AI Engineering API!"}`

---

## LLM Services (Legacy/Direct)

### Text Generation
- **POST** `/summarize` - Summarize text using local LLM
  - Request: `{"text": "string"}`
  - Response: `SummarizationResponse`

- **POST** `/generate` - Generate text using local LLM
  - Request: `{"text": "string"}`
  - Response: `GenerationResponse`

- **POST** `/sentiment` - Classify sentiment using local LLM
  - Request: `{"text": "string"}`
  - Response: `SentimentResponse`

### External LLM Services
- **POST** `/generate/openai` - Generate text using OpenAI API
  - Request: `{"text": "string"}`
  - Response: `GenerationResponse`

- **POST** `/generate/hf` - Generate text using Hugging Face API
  - Request: `{"text": "string"}`
  - Response: `GenerationResponse`

- **POST** `/generate/ideas` - Generate content ideas using LangChain
  - Request: `IdeaRequest`
  - Response: `IdeaResponse`

- **POST** `/chat` - Conversational chat endpoint
  - Request: `ChatRequest`
  - Response: `ChatResponse`

### Legacy RAG Endpoints (in main.py)
- **POST** `/ask-document` - Ask question against documents (simple)
  - Request: `{"text": "string"}`
  - Response: `GenerationResponse`

- **POST** `/add-document` - Add document via file upload (simple)
  - Request: File upload
  - Response: `GenerationResponse`

---

## RAG Services (Primary - `/api/rag/*`)

### Query & Retrieval
- **POST** `/api/rag/{model_provider}/query` - Main RAG query endpoint
  - **Path Parameter**: `model_provider` - `"local"` or `"google"`
  - **Headers**: 
    - `X-API-Key` (optional) - For RBAC authentication
    - `X-Session-Id` (optional) - For session-aware queries
  - **Request Body**:
    ```json
    {
      "question": "string",
      "top_k": 3,
      "use_llm": false,
      "max_tokens": 256,
      "category": "optional_string"
    }
    ```
  - **Response**: `QueryResponse` with answer, retrieved documents, and context
  - **Features**:
    - RBAC filtering based on user role/department
    - Session-aware with conversation history
    - Onboarding flow support
    - Tone-aware responses
    - Multi-turn chat support

### Document Management
- **POST** `/api/rag/add` - Add document via JSON
  - **Headers**: `X-API-Key` (optional)
  - **Request Body**:
    ```json
    {
      "source_name": "string",
      "text": "string",
      "metadata": {
        "department": "string",
        "sensitivity": "public_internal|department_confidential|role_confidential|highly_confidential|personal",
        "allowed_roles": ["string"],
        "owner_id": "string"
      }
    }
    ```
  - **Response**: `AddResponse` with message and chunk count

- **POST** `/api/rag/add-file` - Upload and add document file
  - **Headers**: `X-API-Key` (optional)
  - **Query Parameters**:
    - `department` (optional, default: "General")
    - `sensitivity` (optional, default: "public_internal")
  - **Request**: Multipart file upload (max 5MB)
  - **Supported Formats**: `.md`, `.markdown`, `.html`, `.htm`, `.json`, `.txt`
  - **Response**: `AddResponse` with message and chunk count

- **POST** `/api/rag/seed` - Seed database with default documents
  - **Headers**: `X-API-Key` (optional)
  - **Query Parameters**:
    - `reseed` (optional, default: false) - Force re-seeding
  - **Response**: `AddResponse` with message and chunk count
  - **Note**: Seeds from `data/companyData` directory

- **POST** `/api/rag/clear` - Clear all documents from collection
  - **Headers**: `X-API-Key` (required)
  - **Authorization**: Executive or Legal roles only
  - **Response**: `AddResponse` with confirmation message

### Session Management
- **POST** `/api/rag/session/start` - Start a support chat session
  - **Headers**: `X-API-Key` (optional)
  - **Response**: 
    ```json
    {
      "session_id": "string",
      "message": "Session started"
    }
    ```

- **POST** `/api/rag/session/end` - End a support chat session
  - **Headers**: `X-API-Key` (optional)
  - **Request Body**:
    ```json
    {
      "session_id": "string"
    }
    ```
  - **Response**: 
    ```json
    {
      "session_id": "string",
      "message": "Support session ended."
    }
    ```

### Sentiment Analysis
- **POST** `/api/rag/sentiment` - Analyze sentiment and tone of text
  - **Request Body**:
    ```json
    {
      "text": "string"
    }
    ```
  - **Response**:
    ```json
    {
      "ok": true,
      "result": {
        "text": "string",
        "sentiment": "positive|negative|neutral|unknown",
        "tone": "angry|confused|happy|frustrated|polite|urgent|neutral",
        "proba": {
          "sentiment": {"positive": 0.8, "negative": 0.1, "neutral": 0.1},
          "tone": {"polite": 0.7, "neutral": 0.3}
        }
      }
    }
    ```

- **GET** `/api/rag/sentiment/stats` - Get sentiment statistics
  - **Response**: Sentiment statistics from stored sessions

---

## Authentication & Authorization

### API Key Authentication
- **Header**: `X-API-Key`
- **Roles**: Employee, Manager, HR, Legal, Executive, Guest (no key)
- **Departments**: Engineering, Finance, HR, Legal, IT, Executive, General

### RBAC Sensitivity Levels
1. `public_internal` - All authenticated users
2. `department_confidential` - Same department or HR/Legal/Executive
3. `role_confidential` - Specific roles or HR/Legal/Executive
4. `highly_confidential` - Legal/Executive only
5. `personal` - Owner or HR/Legal/Executive

---

## Notes for Next Requirements

### Current Architecture
- **Modular RAG Service**: Functions decomposed for testability
- **Dependency Injection**: Services injected via `Depends(get_rag_service)`
- **Session Support**: Multi-turn conversations with history
- **Document Parsing**: Supports Markdown, HTML, JSON, plain text
- **Async Operations**: All RAG operations are asynchronous

### Potential Improvements
- [ ] Add pagination for document retrieval
- [ ] Add document update/delete endpoints
- [ ] Add bulk document upload
- [ ] Add search/filter endpoints for documents
- [ ] Add user management endpoints
- [ ] Add analytics/metrics endpoints
- [ ] Add document versioning
- [ ] Add export/import functionality

---

**Last Updated**: 2025-11-25
