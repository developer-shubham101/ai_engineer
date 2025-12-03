# 🚀 Multi-Provider Enterprise RAG System - Project Overview

## Project Goals & Vision

This project is an **enterprise-grade RAG system** with multi-provider LLM support, advanced RBAC, and offline-first architecture. It serves as a production-ready **Retrieval-Augmented Generation (RAG) system** that works with multiple LLM providers through a unified API.

## Core Objectives

### Primary Goals
- Provide quick policy and IT-related answers to employees without waiting for HR or IT representatives
- Centralize knowledge retrieval with a robust RAG pipeline
- Ensure sensitive information is protected through access-level filtering
- Enable departments (HR, Finance, IT) to update and version their own documents
- Offer an external chatbot for public-facing content

### Secondary Goals
- Ensure future compatibility with cloud-based LLM APIs
- Maintain audit logs for updates and user interactions
- Provide administrative tools for document ingestion and metadata tagging

## Key Features

- 🤖 **Multi-Provider Support** - Local models, OpenAI GPT, Google Gemini, Hugging Face
- 🔒 **Enterprise RBAC** - Role-based access control with flexible overrides
- 📚 **Document Versioning** - Non-destructive updates with full history
- 💬 **Session Management** - Persistent conversations with context
- 🚀 **Offline-First** - Works without internet using local LLMs
- ⚡ **Optimized Prompts** - Smart token budgeting and context truncation
- 🔍 **Debug Tools** - Complete prompt/response logging
- 🛡️ **Security** - JWT authentication with audit trails

## System Architecture

### High-Level Components
1. **Chat Interface (Internal & Public)**
2. **RAG Engine**
   - Document Store (raw files)
   - Chroma Vector Database
   - Embedding Generator (CPU-friendly)
   - Retrieval Layer
3. **Local LLM Engine**
4. **Role-Based Access Authorizer**
5. **Document Management Dashboard**
6. **User Authentication System**
7. **Ticketing System Integration (API or DB)**
8. **Logging & Monitoring Layer**

### Supported Providers

| Provider | Models | Status |
|----------|--------|---------|
| **Local** | Mistral-7B, Phi-2, Llama-3.2, Gemma-2B | ✅ Offline |
| **OpenAI** | GPT-3.5, GPT-4 | ✅ API |
| **Google** | Gemini-2.5-Flash, Gemini-2.5-Pro | ✅ API |
| **Hugging Face** | Various models | ✅ API |

## Role-Based Access Control (RBAC)

### User Roles
- **Public**
- **Employee L1**
- **Employee L2**
- **Team Lead**
- **Manager**
- **HR**
- **Finance**
- **IT Admin**
- **Super Admin**

### Role Hierarchy
```
SuperAdmin (Level 4) ──┐
                       ├── Full system access
Manager (Level 3) ─────┤
                       ├── Department management
HR (Level 2) ──────────┤
                       ├── Employee data access
Employee (Level 1) ────┤
                       ├── Standard documents
Guest (Level 0) ───────┘
                       └── Public content only
```

## Use Cases

This system is perfect for:

- 🏢 **Enterprise Knowledge Management** - Centralized company information with role-based access
- 🎓 **Learning RAG Systems** - Complete implementation with multiple providers
- 🔬 **AI Research** - Experiment with different LLM providers and prompt strategies
- 🛠️ **Prototyping** - Quick setup for AI-powered applications
- 📚 **Document Q&A** - Intelligent search and retrieval from document collections

## Implementation Phases

### Phase 1 — Foundation
- Setup Chroma, embeddings, LLM runtime
- Build ingestion + vectorization pipeline

### Phase 2 — Internal Chatbot
- Implement RBAC middleware
- Add policy/IT document retrieval
- Integrate ticket lookup

### Phase 3 — Dashboard Development
- Document management UI
- Versioning system
- Department-level permissions

### Phase 4 — Public Chatbot
- Deploy public-safe content pipeline
- Implement website widget

### Phase 5 — Hardening & Scaling
- Logging, analytics, monitoring
- Add fallback APIs if required

## Security & Privacy

### Authentication
- SSO (Azure AD / Google Workspace / LDAP)
- JWT-based access tokens for API usage

### Authorization
- Attribute-based access control (ABAC)
- Each metadata field validates permissions

### Policy Compliance
- GDPR-aligned storage
- No external APIs used by default
- Logs anonymized for analytics

## Technical Requirements

### Prerequisites
- Python 3.8+
- 8GB+ RAM (for local models)
- Git

### Environment Variables
```bash
 
# Server configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false

# Model settings
DEFAULT_MODEL_NAME=mistral-7b-instruct-v0.2
EMBEDDING_MODEL_NAME=bge-small-en-v1.5
```

## Project Structure


This system provides a secure, scalable, and efficient internal knowledge platform using a local RAG pipeline. It ensures sensitive information is appropriately protected while giving employees and external users fast, intelligent access to company knowledge.