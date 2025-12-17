# 🔌 API Documentation & Testing Guide

## Overview

This document provides comprehensive API documentation and testing examples for the Multi-Provider Enterprise RAG System. The system provides both REST API endpoints and interactive testing capabilities, including new multimodal AI processing capabilities.

**Base URL:** `http://localhost:8000` (or your configured host and port)

## 📑 Table of Contents

1. [General Endpoints](#-general-endpoints)
2. [Authentication & Authorization](#-authentication--authorization)
3. [Multimodal AI Processing](#-multimodal-ai-processing-new)
   - [Audio Processing APIs](#-audio-processing-apis)
   - [Vision Processing APIs](#-vision-processing-apis)
   - [Media File Serving](#-media-file-serving)
   - [Complete Multimodal Workflows](#-complete-multimodal-workflows)
4. [Conversation History](#-conversation-history-new)
5. [Prompt Templates](#-prompt-templates-new)
6. [RAG Query APIs](#-rag-query-apis)
7. [Document Management](#-document-management)
8. [Model Management](#-model-management)

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

## 🎭 Multimodal AI Processing (NEW)

### Overview
The system now supports multimodal AI capabilities including:
- **Speech-to-Text (STT)**: Convert audio to text using Vosk or Whisper
- **Text-to-Speech (TTS)**: Convert text to audio using pyttsx3 or espeak
- **Optical Character Recognition (OCR)**: Extract text from images using Tesseract or PaddleOCR
- **Image Analysis**: Basic image description and analysis
- **Emotion Detection**: Detect emotions from audio using audio feature analysis
- **Secure File Management**: User-isolated file storage with RBAC

### File Storage Structure
```
user_uploaded_files/
├── user_123/
│   ├── audio_conv_456_001.wav    # STT input
│   ├── tts_conv_456_002.mp3      # TTS output  
│   ├── image_conv_456_003.jpg    # Vision input
│   └── doc_conv_456_004.pdf      # OCR input
```

## 🎙️ Audio Processing APIs

### Speech-to-Text (STT)
- **POST** `/api/audio/stt` - Convert speech to text

```bash
curl -X POST "http://localhost:8000/api/audio/stt" \
-H "Authorization: Bearer $TOKEN" \
-F "file=@voice_question.wav" \
-F "provider=vosk" \
-F "conversation_id=conv_123"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "text": "What is our company vacation policy?",
    "provider": "vosk",
    "confidence": 0.8
  },
  "file_path": "user_uploaded_files/user123/audio_conv123_1640995200.wav",
  "error": null
}
```

**Supported Providers:**
- `vosk` (default): CPU-friendly, offline, good accuracy
- `whisper`: Higher accuracy, slower processing

### Text-to-Speech (TTS)
- **POST** `/api/audio/tts` - Convert text to speech

```bash
curl -X POST "http://localhost:8000/api/audio/tts" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "text": "Here is your answer about the vacation policy",
  "conversation_id": "conv_123",
  "provider": "pyttsx3"
}'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "text": "Here is your answer about the vacation policy",
    "provider": "pyttsx3",
    "duration": 5.2
  },
  "file_path": "user_uploaded_files/user123/tts_conv123_1640995201.wav",
  "error": null
}
```

**Supported Providers:**
- `pyttsx3` (default): Cross-platform, offline
- `espeak`: Lightweight, command-line based

### Emotion Detection
- **POST** `/api/audio/emotion` - Detect emotion from audio

```bash
curl -X POST "http://localhost:8000/api/audio/emotion" \
-H "Authorization: Bearer $TOKEN" \
-F "file=@voice_sample.wav" \
-F "provider=basic" \
-F "conversation_id=conv_123"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "emotion": "positive",
    "confidence": 0.7,
    "provider": "basic",
    "features": {
      "mean_mfcc": 0.15,
      "mean_spectral": 1500.0,
      "mean_zcr": 0.08
    }
  },
  "file_path": "user_uploaded_files/user123/audio_conv123_1640995202.wav",
  "error": null
}
```

**Detected Emotions:**
- `excited`: High spectral centroid + high zero crossing rate
- `calm`: Low spectral centroid
- `positive`: Positive MFCC features
- `neutral`: Default classification

## 👁️ Vision Processing APIs

### Optical Character Recognition (OCR)
- **POST** `/api/vision/ocr` - Extract text from images

```bash
curl -X POST "http://localhost:8000/api/vision/ocr" \
-H "Authorization: Bearer $TOKEN" \
-F "file=@document_scan.jpg" \
-F "provider=tesseract" \
-F "conversation_id=conv_123"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "text": "Company Policy Document\n\nVacation Policy:\nEmployees are entitled to 20 days of annual leave...",
    "provider": "tesseract",
    "confidence": 0.8
  },
  "file_path": "user_uploaded_files/user123/image_conv123_1640995203.jpg",
  "error": null
}
```

**Supported Providers:**
- `tesseract` (default): Widely supported, good for printed text
- `paddleocr`: Better accuracy, supports multiple languages

### Image Description
- **POST** `/api/vision/describe` - Generate basic image description

```bash
curl -X POST "http://localhost:8000/api/vision/describe" \
-H "Authorization: Bearer $TOKEN" \
-F "file=@photo.jpg" \
-F "provider=tesseract" \
-F "conversation_id=conv_123"
```

**Response:**
```json
{
  "success": true,
  "data": {
    "description": "Image: 1920x1080 pixels, RGB mode",
    "provider": "basic",
    "width": 1920,
    "height": 1080,
    "mode": "RGB"
  },
  "file_path": "user_uploaded_files/user123/image_conv123_1640995204.jpg",
  "error": null
}
```

### Image Analysis
- **POST** `/api/vision/analyze` - Analyze image (currently same as describe)

```bash
curl -X POST "http://localhost:8000/api/vision/analyze" \
-H "Authorization: Bearer $TOKEN" \
-F "file=@image.jpg" \
-F "provider=tesseract" \
-F "conversation_id=conv_123"
```

## 📁 Media File Serving

### Serve Media Files
- **GET** `/api/media/{user_id}/{filename}` - Serve uploaded media files with RBAC

```bash
# Serve audio file
curl "http://localhost:8000/api/media/user123/tts_conv123_1640995201.wav" \
-H "Authorization: Bearer $TOKEN"

# Serve image file
curl "http://localhost:8000/api/media/user123/image_conv123_1640995203.jpg" \
-H "Authorization: Bearer $TOKEN"
```

**Security Features:**
- Users can only access their own files
- File path validation prevents directory traversal
- Proper media type headers for different file types
- Returns 403 Forbidden for unauthorized access
- Returns 404 Not Found for non-existent files

**Supported Media Types:**
- Audio: `.mp3`, `.wav`, `.ogg` → `audio/mpeg`
- Images: `.jpg`, `.jpeg`, `.png`, `.gif` → `image/jpeg`
- Documents: `.pdf` → `application/pdf`
- Other: `application/octet-stream`

## 🔄 Complete Multimodal Workflows

### Voice-to-Voice Conversation
```bash
# 1. Convert voice question to text
STT_RESPONSE=$(curl -X POST "http://localhost:8000/api/audio/stt" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@question.wav" \
  -F "conversation_id=conv_123")

# 2. Extract text from response
QUESTION_TEXT=$(echo $STT_RESPONSE | jq -r '.data.text')

# 3. Query RAG system with extracted text
RAG_RESPONSE=$(curl -X POST "http://localhost:8000/api/rag/local/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"question\": \"$QUESTION_TEXT\", \"conversation_id\": \"conv_123\", \"use_llm\": true}")

# 4. Convert AI answer to speech
ANSWER_TEXT=$(echo $RAG_RESPONSE | jq -r '.answer')
TTS_RESPONSE=$(curl -X POST "http://localhost:8000/api/audio/tts" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"$ANSWER_TEXT\", \"conversation_id\": \"conv_123\"}")

# 5. Get audio file path
AUDIO_PATH=$(echo $TTS_RESPONSE | jq -r '.file_path')
echo "Audio response saved to: $AUDIO_PATH"
```

### Document OCR to RAG Workflow
```bash
# 1. Extract text from scanned document
OCR_RESPONSE=$(curl -X POST "http://localhost:8000/api/vision/ocr" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@scanned_policy.jpg" \
  -F "conversation_id=conv_123")

# 2. Extract text from OCR response
EXTRACTED_TEXT=$(echo $OCR_RESPONSE | jq -r '.data.text')

# 3. Add extracted text to RAG document store
curl -X POST "http://localhost:8000/api/rag/documents/add" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d "{
    \"source_name\": \"Scanned Policy Document\",
    \"text\": \"$EXTRACTED_TEXT\",
    \"metadata\": {
      \"sensitivity\": \"public_internal\",
      \"department\": \"HR\",
      \"source_type\": \"ocr_scan\"
    }
  }"

# 4. Query the newly added document
curl -X POST "http://localhost:8000/api/rag/local/query" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does the scanned document say about vacation policy?",
    "conversation_id": "conv_123",
    "use_llm": true
  }'
```

### Error Handling

All multimodal APIs return consistent error responses:

```json
{
  "success": false,
  "data": {},
  "file_path": null,
  "error": "Specific error message"
}
```

**Common Error Cases:**
- **Missing Dependencies**: "Vosk not installed. Run: pip install vosk"
- **Invalid File Type**: "File must be audio format"
- **Model Not Found**: "Vosk model not found. Please download vosk-model-small-en-us-0.15"
- **Processing Failed**: Specific error from underlying library
- **Access Denied**: "Access denied" (403 status)
- **File Not Found**: "File not found" (404 status)

### Installation Requirements

To use multimodal features, install additional dependencies:

```bash
# Install multimodal requirements
pip install -r requirements_multimodal.txt

# Or install individually:
pip install vosk openai-whisper pyttsx3 pytesseract Pillow paddlepaddle paddleocr librosa soundfile
```

**System Dependencies:**
- **Tesseract OCR**: Install system package (e.g., `apt install tesseract-ocr` on Ubuntu)
- **espeak**: Install system package (e.g., `apt install espeak-ng` on Ubuntu)
- **Vosk Models**: Download to `models/vosk-model-small-en-us-0.15/`

## 💬 Conversation History (NEW)

### Overview
The system now supports persistent conversation history that is tied to user accounts rather than sessions. This enables ChatGPT-like conversation management where users can:
- Access conversation history across different devices
- View and restore previous conversations
- Continue conversations from where they left off
- Full RAG pipeline logging for debugging and analytics

## 📝 Prompt Templates (NEW)

### Overview
Manage dynamic prompt templates stored in the database. These templates are used by the RAG orchestrator to generate context-aware prompts.

### List Templates
- **GET** `/api/templates` - List all templates

```bash
curl -X GET "http://localhost:5444/api/templates" \
-H "Authorization: Bearer $TOKEN"
```

### Create Template
- **POST** `/api/templates` - Create a new template

```bash
curl -X POST "http://localhost:5444/api/templates" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "name": "creative_chat",
  "content": "System: You are a creative assistant.\n\nContext: {source_docs}\n\nQuestion: {user_question}"
}'
```

### Get Specific Template
- **GET** `/api/templates/{name}` - Get template details

```bash
curl -X GET "http://localhost:5444/api/templates/personalized_chat" \
-H "Authorization: Bearer $TOKEN"
```

### Update Template
- **PUT** `/api/templates/{name}` - Update template content

```bash
curl -X PUT "http://localhost:5444/api/templates/personalized_chat" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "content": "System: Updated system prompt...\n\nContext: {source_docs}\n\nQuestion: {user_question}"
}'
```

### Delete Template
- **DELETE** `/api/templates/{name}` - Delete a template

```bash
curl -X DELETE "http://localhost:5444/api/templates/old_template" \
-H "Authorization: Bearer $TOKEN"
```

### List Conversations
- **GET** `/api/conversations` - List all conversations for the authenticated user

```bash
curl -X GET "http://localhost:5444/api/conversations?limit=50&offset=0" \
-H "Authorization: Bearer $TOKEN"
```

**Response:**
```json
[
  {
    "id": "conv_abc123...",
    "user_id": "u_admin_1",
    "title": "Company Leave Policy Discussion",
    "created_at": "2025-12-14T08:00:00Z",
    "updated_at": "2025-12-14T08:15:00Z",
    "message_count": 12
  },
  {
    "id": "conv_def456...",
    "user_id": "u_admin_1",
    "title": "New Conversation",
    "created_at": "2025-12-14T09:00:00Z",
    "updated_at": "2025-12-14T09:00:00Z",
    "message_count": 0
  }
]
```

### Create New Conversation
- **POST** `/api/conversations` - Create a new conversation

```bash
curl -X POST "http://localhost:5444/api/conversations" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{"title": "HR Policy Questions"}'
```

### Get Specific Conversation
- **GET** `/api/conversations/{conversation_id}` - Get conversation details

```bash
curl -X GET "http://localhost:5444/api/conversations/conv_abc123..." \
-H "Authorization: Bearer $TOKEN"
```

### Update Conversation
- **PUT** `/api/conversations/{conversation_id}` - Update conversation (e.g., rename)

```bash
curl -X PUT "http://localhost:5444/api/conversations/conv_abc123..." \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{"title": "Updated Conversation Title"}'
```

### Delete Conversation
- **DELETE** `/api/conversations/{conversation_id}` - Delete conversation (soft delete)

```bash
curl -X DELETE "http://localhost:5444/api/conversations/conv_abc123..." \
-H "Authorization: Bearer $TOKEN"
```

### Get Conversation Messages
- **GET** `/api/conversations/{conversation_id}/messages` - Get all messages with full RAG logging

```bash
curl -X GET "http://localhost:5444/api/conversations/conv_abc123.../messages?limit=50" \
-H "Authorization: Bearer $TOKEN"
```

**Response includes comprehensive RAG logging:**
```json
[
  {
    "id": 1,
    "speaker": "user",
    "content": "What is the company leave policy?",
    "created_at": "2025-12-14T08:00:00Z",
    "sentiment": "neutral",
    "tone": "polite"
  },
  {
    "id": 2,
    "speaker": "assistant",
    "content": "The company provides 20 days of annual leave...",
    "created_at": "2025-12-14T08:00:05Z",
    
    // RAG Pipeline Logging
    "user_query": "What is the company leave policy?",
    "retrieved_context": [
      {
        "id": "doc_123",
        "text": "Leave policy document excerpt...",
        "metadata": {"department": "HR"},
        "distance": 0.85
      }
    ],
    "embeddings_used": {
      "model": "BAAI/bge-small-en-v1.5",
      "dimensions": 384
    },
    "llm_prompt": "System: You are an HR assistant...\n\nContext: Leave policy...\n\nQuestion: What is the company leave policy?",
    "llm_response_raw": "{\"answer\": \"The company provides...\", \"sources\": [...]}",
    "llm_provider": "local",
    "llm_model": "phi2",
    "llm_tokens_used": null,
    "llm_temperature": 0.1,
    "llm_max_tokens": 256,
    "retrieved_doc_ids": ["doc_123", "doc_456"],
    "retrieval_top_k": 3,
    "use_documents": true,
    "use_llm": true,
    "processing_time_ms": 1250,
    "error_message": null
  }
]
```

### Restore Conversation
- **POST** `/api/conversations/{conversation_id}/restore` - Restore conversation to current session

```bash
curl -X POST "http://localhost:5444/api/conversations/conv_abc123.../restore" \
-H "Authorization: Bearer $TOKEN"
```

### RAG Logging Fields

Every assistant message in a conversation includes comprehensive RAG pipeline logging:

| Field | Type | Description |
|-------|------|-------------|
| `user_query` | string | Original user question |
| `retrieved_context` | array | Retrieved documents with metadata |
| `embeddings_used` | object | Embedding model information |
| `llm_prompt` | string | Final prompt sent to LLM |
| `llm_response_raw` | string | Raw LLM response |
| `llm_provider` | string | Provider used (local, google, hf) |
| `llm_model` | string | Model name |
| `llm_tokens_used` | integer | Token count (if available) |
| `llm_temperature` | float | Temperature parameter |
| `llm_max_tokens` | integer | Max tokens parameter |
| `retrieved_doc_ids` | array | Document IDs retrieved |
| `retrieval_top_k` | integer | Top K parameter |
| `use_documents` | boolean | Whether documents were used |
| `use_llm` | boolean | Whether LLM was used |
| `processing_time_ms` | integer | Total processing time |
| `error_message` | string | Error message if query failed |

### Use Cases

####### Cross-Device Access
```bash
# Login from Device A
curl -X POST "http://localhost:8000/api/auth/token" \
-H "Content-Type: application/json" \
-d '{"username": "admin", "password": "admin123"}'

# Have a conversation...

# Login from Device B with same credentials
curl -X POST "http://localhost:8000/api/auth/token" \
-H "Content-Type: application/json" \
-d '{"username": "admin", "password": "admin123"}'

# List conversations - shows all conversations from Device A
curl -X GET "http://localhost:8000/api/conversations" \
-H "Authorization: Bearer $TOKEN"
```

## 🤖 RAG Query APIs

### Multi-Provider Query Interface
The system supports multiple LLM providers through a unified API interface.

**Supported Providers:**
- `local` - Local models (Mistral-7B, Phi-2, Llama-3.2, etc.)
- `google` - Google Gemini API
- `gpt` - OpenAI GPT API
- `huggingface` or `hf` - Hugging Face Inference API

### Query with Local Models
- **POST** `/api/rag/local/query` - Query using local LLM models

```bash
curl -X POST "http://localhost:8000/api/rag/local/query" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "question": "What is our company vacation policy?",
  "conversation_id": "conv_123",
  "top_k": 3,
  "use_documents": true,
  "use_llm": true,
  "use_conversation_history": true,
  "enable_agentic_mode": false,
  "max_tokens": 256,
  "temperature": 0.1,
  "prompt_template": "personalized_chat",
  "local_llm_model": "phi2"
}'
```

### Query with Cloud Providers
```bash
# Google Gemini
curl -X POST "http://localhost:8000/api/rag/google/query" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "question": "What is our company vacation policy?",
  "conversation_id": "conv_123",
  "use_llm": true,
  "temperature": 0.1
}'

# OpenAI GPT
curl -X POST "http://localhost:8000/api/rag/gpt/query" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "question": "What is our company vacation policy?",
  "conversation_id": "conv_123",
  "use_llm": true,
  "temperature": 0.1
}'

# Hugging Face
curl -X POST "http://localhost:8000/api/rag/huggingface/query" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "question": "What is our company vacation policy?",
  "conversation_id": "conv_123",
  "use_llm": true,
  "temperature": 0.1
}'
```

### Query Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `question` | string | required | User question |
| `conversation_id` | string | required | Conversation ID for history |
| `top_k` | integer | 3 | Number of documents to retrieve |
| `use_documents` | boolean | true | Whether to retrieve documents |
| `use_llm` | boolean | false | Whether to use LLM for response |
| `use_conversation_history` | boolean | true | Include conversation context |
| `enable_agentic_mode` | boolean | false | Enable step-by-step reasoning |
| `max_tokens` | integer | 256 | Maximum response tokens |
| `temperature` | float | 0.1 | Response creativity (0.0-1.0) |
| `category` | string | null | Document category filter |
| `debug` | boolean | false | Include debug information |
| `prompt_template` | string | "" | Template name to use |
| `local_llm_model` | string | null | Specific local model (local provider only) |

### Response Format

```json
{
  "answer": "The company provides 20 days of annual leave per year...",
  "retrieved": [
    {
      "id": "doc_123",
      "text": "Leave policy document excerpt...",
      "metadata": {
        "department": "HR",
        "sensitivity": "public_internal",
        "source": "HR_policies_handbook.md"
      },
      "distance": 0.85
    }
  ],
  "context": "Combined context from retrieved documents...",
  "final_prompt": "System: You are an HR assistant...\n\nContext: Leave policy...\n\nQuestion: What is our company vacation policy?"
}
```

### Agentic Mode (NEW)

When `enable_agentic_mode` is set to `true`, the system enhances the prompt with reasoning instructions:

```bash
curl -X POST "http://localhost:8000/api/rag/local/query" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "question": "What are the steps to request vacation time?",
  "conversation_id": "conv_123",
  "use_llm": true,
  "enable_agentic_mode": true
}'
```

**Agentic Mode Benefits:**
- Step-by-step reasoning in responses
- More detailed explanations
- Explicit logic chains
- Better for complex analysis

## 📄 Document Management

### Add Document (JSON)
- **POST** `/api/rag/documents/add` - Add document via JSON

```bash
curl -X POST "http://localhost:8000/api/rag/documents/add" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "source_name": "New Company Policy",
  "text": "This document outlines the new remote work policy...",
  "metadata": {
    "department": "HR",
    "sensitivity": "public_internal",
    "document_type": "policy"
  }
}'
```

### Upload Document File
- **POST** `/api/rag/documents/add-file` - Upload and add document file

```bash
curl -X POST "http://localhost:8000/api/rag/documents/add-file" \
-H "Authorization: Bearer $TOKEN" \
-F "file=@policy_document.pdf" \
-F "department=HR" \
-F "sensitivity=public_internal"
```

### List Documents
- **GET** `/api/rag/documents/list` - List documents with filtering

```bash
curl -X GET "http://localhost:8000/api/rag/documents/list?department=HR&status=published&latest_only=true" \
-H "Authorization: Bearer $TOKEN"
```

### Document Versioning

```bash
# Update document (creates new version)
curl -X POST "http://localhost:8000/api/rag/documents/update" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer $TOKEN" \
-d '{
  "document_id": "doc_123",
  "text": "Updated policy content...",
  "version_notes": "Updated remote work guidelines",
  "status": "published"
}'

# Get version history
curl -X GET "http://localhost:8000/api/rag/documents/doc_123/versions" \
-H "Authorization: Bearer $TOKEN"

# Compare versions
curl -X GET "http://localhost:8000/api/rag/documents/doc_123/compare?version1=v1.0&version2=v2.0" \
-H "Authorization: Bearer $TOKEN"
```

### Seed Sample Documents
- **POST** `/api/rag/documents/seed` - Load sample company documents

```bash
curl -X POST "http://localhost:8000/api/rag/documents/seed?reseed=false" \
-H "Authorization: Bearer $TOKEN"
```

## 🤖 Model Management

### List Available Models
- **GET** `/api/models/list` - List all available local models

```bash
curl -X GET "http://localhost:8000/api/models/list" \
-H "Authorization: Bearer $TOKEN"
```

**Response:**
```json
{
  "models": [
    {
      "key": "phi2",
      "name": "Phi-2 Q4_K_M",
      "file_path": "models/phi-2-q4_k_m.gguf",
      "size_gb": 1.6,
      "context_length": 2048,
      "recommended_use": "General Q&A, reasoning tasks"
    },
    {
      "key": "mistral7b",
      "name": "Mistral-7B Instruct v0.2",
      "file_path": "models/mistral-7b-instruct-v0.2.Q3_K_M.gguf",
      "size_gb": 3.8,
      "context_length": 4096,
      "recommended_use": "Production RAG, instruction following"
    }
  ],
  "count": 2,
  "default_model": "phi2"
}
```

### Get Best Available Model
- **GET** `/api/models/best` - Get the best available model for current system

```bash
curl -X GET "http://localhost:8000/api/models/best" \
-H "Authorization: Bearer $TOKEN"
```

### Refresh Model Cache
- **POST** `/api/models/refresh` - Refresh the model cache (scan for new models)

```bash
curl -X POST "http://localhost:8000/api/models/refresh" \
-H "Authorization: Bearer $TOKEN"
```

## 🔧 System Status

### Embedding Model Status
- **GET** `/api/rag/embedding/status` - Check embedding model status

```bash
curl -X GET "http://localhost:8000/api/rag/embedding/status" \
-H "Authorization: Bearer $TOKEN"
```

### Module Status
- **GET** `/api/modules/status` - Check modular architecture status

```bash
curl -X GET "http://localhost:8000/api/modules/status"
```

## 🧪 Testing & Examples

### Complete RAG Workflow Test
```bash
#!/bin/bash

# 1. Login and get token
TOKEN_RESPONSE=$(curl -s -X POST "http://localhost:8000/api/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}')

TOKEN=$(echo $TOKEN_RESPONSE | jq -r '.access_token')
echo "Token: $TOKEN"

# 2. Create new conversation
CONV_RESPONSE=$(curl -s -X POST "http://localhost:8000/api/conversations" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"title": "API Test Conversation"}')

CONV_ID=$(echo $CONV_RESPONSE | jq -r '.id')
echo "Conversation ID: $CONV_ID"

# 3. Query RAG system
RAG_RESPONSE=$(curl -s -X POST "http://localhost:8000/api/rag/local/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d "{
    \"question\": \"What is our company vacation policy?\",
    \"conversation_id\": \"$CONV_ID\",
    \"use_llm\": true,
    \"use_documents\": true
  }")

echo "RAG Response:"
echo $RAG_RESPONSE | jq '.answer'

# 4. Get conversation messages
MESSAGES=$(curl -s -X GET "http://localhost:8000/api/conversations/$CONV_ID/messages" \
  -H "Authorization: Bearer $TOKEN")

echo "Message count: $(echo $MESSAGES | jq 'length')"
```

### Multimodal Workflow Test
```bash
#!/bin/bash

# Assuming TOKEN and CONV_ID from previous example

# 1. Test TTS (Text to Speech)
TTS_RESPONSE=$(curl -s -X POST "http://localhost:8000/api/audio/tts" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN" \
  -d "{
    \"text\": \"Hello, this is a test of the text to speech system\",
    \"conversation_id\": \"$CONV_ID\",
    \"provider\": \"pyttsx3\"
  }")

AUDIO_FILE=$(echo $TTS_RESPONSE | jq -r '.file_path')
echo "Audio file created: $AUDIO_FILE"

# 2. Test OCR (if you have an image file)
# OCR_RESPONSE=$(curl -s -X POST "http://localhost:8000/api/vision/ocr" \
#   -H "Authorization: Bearer $TOKEN" \
#   -F "file=@test_document.jpg" \
#   -F "conversation_id=$CONV_ID")
# 
# echo "OCR Result:"
# echo $OCR_RESPONSE | jq '.data.text'
```

## 🚨 Error Handling

### Common HTTP Status Codes
- `200 OK` - Successful request
- `400 Bad Request` - Invalid request parameters
- `401 Unauthorized` - Missing or invalid authentication
- `403 Forbidden` - Insufficient permissions (RBAC)
- `404 Not Found` - Resource not found
- `413 Payload Too Large` - File too large
- `500 Internal Server Error` - Server error

### Error Response Format
```json
{
  "detail": "Specific error message",
  "status_code": 400,
  "error_type": "validation_error"
}
```

### RBAC Error Examples
```json
{
  "detail": "Your role 'Employee' (level 1) cannot create documents with sensitivity 'highly_confidential' (requires level 3+)",
  "status_code": 403
}
```

## 📝 Notes

- All timestamps are in ISO 8601 format (UTC)
- File uploads have a maximum size limit (configurable)
- RBAC filtering is applied automatically based on user role and department
- Conversation history is persistent across sessions
- Multimodal files are automatically cleaned up after 7 days
- Debug mode (`debug: true`) provides additional response details
- Temperature parameter controls response creativity (0.0 = deterministic, 1.0 = creative)
- Agentic mode provides step-by-step reasoning in responses

---

**Last Updated:** 2025-01-11  
**API Version:** 1.0.0  
**Documentation Version:** 2.0.0 (includes multimodal features)ization: Bearer $NEW_TOKEN"
```

#### Analytics and Debugging
```bash
# Get messages with full RAG logging for analysis
curl -X GET "http://localhost:5444/api/conversations/conv_abc123.../messages" \
-H "Authorization: Bearer $TOKEN" | \
jq '.[] | select(.speaker == "assistant") | {processing_time_ms, llm_provider, error_message}'
```


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