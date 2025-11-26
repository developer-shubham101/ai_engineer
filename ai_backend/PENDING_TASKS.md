# 📋 Pending Tasks & Feature Gaps

**Date Created:** 2025-11-26  
**Comparison:** [original_gole.md](file:///i:/Workspace/GitHub/ai_engineer/ai_backend/original_gole.md) vs [APP_CONTEXT.md](file:///i:/Workspace/GitHub/ai_engineer/ai_backend/APP_CONTEXT.md)

---

## ✅ Already Implemented

### Core RAG Infrastructure
- ✅ Chroma Vector Database
- ✅ Local embeddings (MiniLM)
- ✅ Local LLM (Mistral 7B)
- ✅ Text chunking
- ✅ Document ingestion (JSON, file upload)
- ✅ Semantic retrieval

### Authentication & Authorization
- ✅ JWT-based authentication
- ✅ Role-Based Access Control (RBAC)
- ✅ User roles: SuperAdmin, HR, Manager, Employee, Guest
- ✅ Sensitivity levels: public_internal, department_confidential, role_confidential, highly_confidential, personal
- ✅ RBAC filtering in retrieval

### API & Integration
- ✅ FastAPI backend
- ✅ Multi-provider support (local, Google)
- ✅ Seed data from companyData
- ✅ Support chat with session management
- ✅ Sentiment analysis

---

## 🚀 Features Beyond Original Specification

**The following features were implemented but NOT part of the original specification:**

### 1. **Google Gemini API Integration** ✅
- **Original Spec:** Listed as "Future Enhancement" (Section 11)
- **Current State:** Fully implemented with `/api/rag/google/query` endpoint
- **Value:** Production-ready cloud LLM option, not just a planned feature

### 2. **Sentiment Analysis System** ✅
- **Original Spec:** Not mentioned
- **Features:**
  - Local sentiment classifier (positive/negative/neutral)
  - Tone detection (angry, confused, happy, frustrated, polite, urgent, neutral)
  - Sentiment stats API (`/api/rag/sentiment/stats`)
  - Integrated into support chat for emotion tracking

### 3. **Guest User Onboarding System** ✅
- **Original Spec:** Not mentioned
- **Features:**
  - Configurable onboarding questions (`onboarding_fields.json`)
  - Progressive profile building during conversation
  - Session-based profile storage
  - Automatic question progression

### 4. **Advanced User Profile System (user_meta)** ✅
- **Original Spec:** Basic user management only
- **Features:**
  - Dynamic key-value profile storage in database
  - Profile-aware chat responses
  - Profile persistence across sessions
  - Profile included in login response

### 5. **Support Chat Session Management** ✅
- **Original Spec:** Not mentioned
- **Features:**
  - SQLite-based session tracking
  - Conversation history storage
  - Session profiles with metadata
  - Tone-aware response generation

### 6. **Document Parser Module** ✅
- **Original Spec:** Only mentioned file formats
- **Features:**
  - Modular, extensible `doc_parser` in `app/utils/`
  - Format detection (Markdown, HTML, JSON, plain text)
  - Easy to extend for new formats
  - Used in file upload endpoint

### 7. **Dependency Injection Architecture** ✅
- **Original Spec:** Not mentioned
- **Features:**
  - `app/dependencies.py` for service injection
  - Better testability and separation of concerns
  - FastAPI `Depends()` pattern throughout

### 8. **Multi-Provider Query Architecture** ✅
- **Original Spec:** Local LLM only initially
- **Features:**
  - Path-based provider selection (`/api/rag/{model_provider}/query`)
  - Unified interface across providers
  - Easy to add new providers (OpenAI, HuggingFace, etc.)

---

## ❌ Missing Features (High Priority)

### 1. Document Management APIs (Backend Only)
**Status:** Partially implemented  
**Original Spec:** Section 6 - Document management functionality

**Missing Backend APIs:**
- ❌ Document update endpoint (with versioning)
- ❌ Document approval workflow API (draft → approved → published)
- ❌ Soft-delete/archive endpoint
- ❌ Version history retrieval API
- ❌ List documents with filtering (by department, status, version)

**Current State:** Basic endpoints exist (`/add`, `/add-file`, `/seed`, `/clear`) but no versioning, approval, or update capabilities

---

### 2. Ticket System Integration
**Status:** Not implemented  
**Original Spec:** Section 7.3 - IT ticket status lookup

**Missing Features:**
- ❌ Integration with ITSM/Jira Service Desk
- ❌ Ticket status queries ("What's the status of my ticket?")
- ❌ Show all open tickets for user
- ❌ Ticket database connection/API

**Impact:** Cannot answer IT support ticket queries

---

### 3. Document Versioning System ✅
**Status:** **IMPLEMENTED (2025-11-26)**  
**Original Spec:** Section 4.1 - Version tracking (e.g., 2025.1, 2026.0)

**Implemented Features:**
- ✅ Version field in metadata (semantic versioning: 1.0, 2.0, 3.0)
- ✅ Version history storage in SQLite database (`version_tracking.py`)
- ✅ Update creates new version (non-destructive)
- ✅ Retrieve specific version via API
- ✅ Compare versions with diff
- ✅ Version metadata: document_id, version, version_created_at, version_created_by, parent_version, status, is_latest_version
- ✅ **Folder-Based Versioning**: Auto-detection from `data/{category}/v{version}/*.md`
- ✅ **Reorganized APIs**: All document endpoints moved to `/api/rag/documents/*`
- ✅ New service functions: `update_document_version()`, `get_document_version()`, `compare_document_versions()`, `list_documents()`, `archive_document_version()`
- ✅ New API endpoints:
  - `POST /api/rag/documents/add` (moved)
  - `POST /api/rag/documents/add-file` (moved)
  - `POST /api/rag/documents/seed` (moved)
  - `POST /api/rag/documents/clear` (moved)
  - `POST /api/rag/documents/update`
  - `GET /api/rag/documents/list`
  - `GET /api/rag/documents/{document_id}/versions`
  - `GET /api/rag/documents/{document_id}/versions/{version}`
  - `GET /api/rag/documents/{document_id}/compare`
  - `POST /api/rag/documents/{document_id}/archive`

**Current State:** Fully functional version system with auto-versioning on add/update

---

### 4. Approval Workflow
**Status:** Not implemented  
**Original Spec:** Section 6.1 - Approve policies before going live

**Missing Features:**
- ❌ Draft/pending status for documents
- ❌ Approval queue
- ❌ Approve/reject actions
- ❌ Publish after approval
- ❌ Role-based approval permissions

---

### 5. Public vs Internal Chatbot Separation
**Status:** Partially implemented  
**Original Spec:** Sections 7 & 8 - Separate public and internal chatbots

**Current Implementation:**
- ✅ Single endpoint handles both (Guest role = public)
- ✅ Public users see only `public_internal` docs

**Missing Features:**
- ❌ Dedicated public chatbot endpoint/widget
- ❌ Dual content model (public summary + sensitive detail)
- ❌ Website widget integration
- ❌ Public content scrubbing/redaction

---

### 6. Audit Logging & Monitoring
**Status:** Basic logging only  
**Original Spec:** Sections 2 & 10.3 - Comprehensive audit logs

**Current Implementation:**
- ✅ Python logging to console

**Missing Features:**
- ❌ User interaction logs (who queried what, when)
- ❌ Document update audit trail
- ❌ Access attempt logs
- ❌ Failed permission logs
- ❌ Anonymized analytics
- ❌ Persistent log storage (DB/files)
- ❌ Audit log retrieval API

---

### 7. SSO Integration
**Status:** Not implemented  
**Original Spec:** Section 10.1 - SSO (Azure AD / Google Workspace / LDAP)

**Current Implementation:**
- ✅ JWT with username/password

**Missing Features:**
- ❌ Azure AD integration
- ❌ Google Workspace SSO
- ❌ LDAP authentication
- ❌ OAuth2 flows

---

### 8. Enhanced Document Metadata
**Status:** Partial implementation  
**Original Spec:** Section 4.1 - Rich metadata schema

**Current Metadata:**
- ✅ department
- ✅ sensitivity
- ✅ source
- ✅ ingested_at
- ✅ ingested_by
- ✅ allowed_roles
- ✅ owner_id

**Missing Metadata:**
- ❌ `document_type` (policy / guideline / announcement / FAQ)
- ❌ `access_level` (L1 / L2 / Manager naming)
- ❌ `version` field
- ❌ Original document reference (for chunks)

---

## ⚠️ Missing Features (Medium Priority)

### 9. Advanced File Format Support
**Original Spec:** Section 4.1 - PDF, DOCX, TXT, MD

**Current Support:**
- ✅ Markdown (.md)
- ✅ HTML (.html)
- ✅ JSON (.json)
- ✅ Plain text (.txt)

**Missing:**
- ❌ PDF parsing
- ❌ DOCX parsing

---

### 10. Data Separation Model
**Status:** Not implemented  
**Original Spec:** Section 9 - Dual content per document

**Missing Features:**
- ❌ Public summary field (separate from full content)
- ❌ Dual storage (public + sensitive versions)
- ❌ Automatic content scrubbing for public view
- ❌ Redaction features

**Current State:** Single content with RBAC filtering

---

### 11. Additional Role Granularity
**Original Spec:** Section 5.1 - More role types

**Current Roles:**
- ✅ SuperAdmin, HR, Manager, Employee, Guest

**Missing Roles:**
- ❌ Employee L1 / L2 distinction
- ❌ Team Lead
- ❌ Finance (as distinct role)
- ❌ IT Admin (as distinct role)

**Note:** Can be added to seed data, but not currently defined

---

### 12. Monitoring & Metrics APIs
**Original Spec:** Section 3.1 - Component #8

**Missing Backend APIs:**
- ❌ System health endpoint (`/api/health`)
- ❌ Query performance metrics endpoint
- ❌ Resource usage API
- ❌ Error rate statistics endpoint
- ❌ Alert webhook integration

---

## 💡 Future Enhancements (Low Priority)

### 13. From Original Spec Section 11
- ⏳ Cloud LLM fallback (OpenAI, Gemini) - **Partially done: Google support exists**
- ⏳ Prompt auditing
- ⏳ Automated policy ingestion from email
- ⏳ Strict redaction for sensitive tokens
- ⏳ GPU deployment option

---

## 📊 Summary

| Category | Total | Implemented | Missing |
|----------|-------|-------------|---------|
| **High Priority** | 8 | 1 | 7 |
| **Medium Priority** | 4 | 0 | 4 |
| **Low Priority** | 5 | 1 | 4 |
| **TOTAL** | 17 | 2 | 15 |

**Recently Completed**:
- ✅ Document Versioning System (High Priority #3) - 2025-11-26

---

## 🎯 Recommended Implementation Order

### Phase 1: Foundation Completion
1. **Document Versioning** - Add version metadata tracking
2. **Enhanced Metadata** - Add `document_type` and `access_level`
3. **Additional Roles** - Define Finance, IT Admin, L1/L2

### Phase 2: Document Management APIs
4. **Document Management Endpoints** - Update, list, archive, version history APIs
5. **Approval Workflow APIs** - Draft → Approval → Published state management
6. **Audit Logging** - Comprehensive activity logging with retrieval API

### Phase 3: Integration & Security
7. **Ticket System Integration** - Connect to ITSM/Jira API
8. **SSO Integration** - Azure AD / Google Workspace
9. **Public Chatbot Endpoint** - Dedicated public API endpoint

### Phase 4: Advanced Features
10. **PDF/DOCX Support** - Advanced file parsing
11. **Data Separation Model** - Dual content model (API-based)
12. **Monitoring & Metrics APIs** - System health & performance endpoints

---

## ❓ Clarifying Questions Needed

*Questions will be asked one by one in separate messages to avoid confusion.*
