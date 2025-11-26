# 📘 Technical Specification Document  
## Internal & Public Chatbot System (RAG-Based Architecture)

**Version:** 1.0  
**Prepared For:** Internal Engineering & IT Teams  
**Prepared By:** AI Systems Design  
**Date:** 2025

---

# 1. Introduction

This document outlines the system architecture, functional requirements, data flow, security considerations, and administrative controls for an AI-driven chatbot application built using a **Retrieval-Augmented Generation (RAG)** pipeline.

The system serves two main user groups:

1. **Internal Employees** — Access to internal policies, IT support info, ticket status, department documentation, and sensitive operations.
2. **Public Users** — Access to general company details, public-facing policies, and job information.

A strict **Role-Based Access Control (RBAC)** model ensures that employees only view documents and content aligned with their role, department, and clearance level.

The project begins with **local, CPU-based LLMs and embeddings**, with optional future expansion toward cloud APIs.

---

# 2. System Goals

### Primary Goals
- Provide quick policy and IT-related answers to employees without waiting for HR or IT representatives.
- Centralize knowledge retrieval with a robust RAG pipeline.
- Ensure sensitive information is protected through access-level filtering.
- Enable departments (HR, Finance, IT) to update and version their own documents.
- Offer an external chatbot for public-facing content.

### Secondary Goals
- Ensure future compatibility with cloud-based LLM APIs.
- Maintain audit logs for updates and user interactions.
- Provide administrative tools for document ingestion and metadata tagging.

---

# 3. System Architecture Overview

## 3.1 High-Level Components
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

---

# 4. Detailed RAG Pipeline

## 4.1 Document Ingestion
- HR, IT, Finance upload documents via dashboard.
- Supported formats: PDF, DOCX, TXT, MD.
- Each document is enriched with metadata:
  - `department:` HR / Finance / IT / General
  - `document_type:` policy / guideline / announcement / FAQ
  - `access_level:` L1 / L2 / Manager / HR_only etc.
  - `sensitivity:` public / internal / confidential
  - `version:` e.g., 2025.1, 2026.0

## 4.2 Text Chunking
- Standard chunking strategy: 512–1024 tokens.
- Chunk metadata references original document + version ID.

## 4.3 Embedding Generation
- CPU-optimized models (e.g., BGE-Small, MiniLM, Instructor-XL-lite).
- Embeddings stored in **Chroma** with metadata.

## 4.4 Retrieval
- Semantic search from Chroma using similarity.
- Filtered by:
  - User role
  - Department
  - Access level
  - Sensitivity flag

## 4.5 LLM Response Generation
- Local LLM (e.g., LLaMA 3 8B/3B CPU-optimized, Mistral 7B CPU quantized).
- Uses a structured prompt:
  - Context provided from retrieved chunks.
  - User query.
  - Access constraints.

---

# 5. Role-Based Access Control (RBAC)

## 5.1 User Roles
- **Public**
- **Employee L1**
- **Employee L2**
- **Team Lead**
- **Manager**
- **HR**
- **Finance**
- **IT Admin**
- **Super Admin**

## 5.2 Access Control Logic
### The system will restrict document retrieval based on:
- `min_access_level` required for the document.
- `department_scope` (e.g., HR documents hidden from non-HR).
- `sensitive = true` — visible only to permitted roles.
- Public users only receive documents tagged `sensitivity: public`.

### Example Filtering Rules
- Finance employees may read sensitive finance docs; HR cannot.
- HR employees may read personal policy documents; Finance cannot.
- L1 cannot see L2-only IT guides.
- Public users cannot see any internal chunk.

---

# 6. Document Management Dashboard

## 6.1 Key Features
- Upload new documents.
- Update existing documents (creates new version).
- Tag documents with metadata.
- Approve new policies before they go live.
- Soft-delete or archive documents.
- View version history.

## 6.2 Permission Mapping
| Department | Permissions |
|-----------|-------------|
| HR | Can edit/upload HR policies only |
| Finance | Can edit/upload finance policies only |
| IT | Can upload IT SOPs, password rules |
| Super Admin | Full document control |

---

# 7. Internal Chatbot Features

## 7.1 Policy Queries
- Leave policy
- Remote work policy
- Privacy policy
- Payroll policy
- Travel reimbursement

## 7.2 IT Support
- Password reset steps
- Laptop troubleshooting
- Access rights
- Software installation rules

## 7.3 Ticket Status Lookup
- Integrate with ticket DB (e.g., ITSM, Jira Service Desk).
- Quick queries:
  - “What’s the status of my laptop repair ticket?”
  - “Show all open tickets assigned to me.”

## 7.4 Sensitive Information Handling
- If retrieved content exceeds user’s access level:
  - Block and inform user.
  - Provide alternative general answer.

---

# 8. Public Chatbot Features

- Company overview.
- Public policies.
- Press releases.
- Public financial summaries (e.g., profit, revenue trends).
  - Sensitive financial figures stored separately.
- Job openings.
- Hiring process FAQs.
- Contact info.

---

# 9. Data Separation Model (Sensitive vs Public)

## 9.1 For Each Document:
Every document has two segments:

1. **Public Content** — scrubbed, high-level, safe to show externally.
2. **Sensitive Content** — detailed internal data for permitted roles only.

## 9.2 Example: Financial Report
- Public:
  - Last year revenue
  - Profit summary
  - General budgeting overview
- Sensitive:
  - Department-wise breakdown
  - Forecasts
  - Internal risk assessment

Only certain roles can see the sensitive portion.

---

# 10. Security & Privacy

## 10.1 Authentication
- SSO (Azure AD / Google Workspace / LDAP)
- JWT-based access tokens for API usage

## 10.2 Authorization
- Attribute-based access control (ABAC)
- Each metadata field validates permissions

## 10.3 Policy Compliance
- GDPR-aligned storage
- No external APIs used by default
- Logs anonymized for analytics

---

# 11. Future Enhancements

1. Allow cloud LLM fallback (OpenAI, Gemini, etc.).
2. Add prompt auditing and monitoring.
3. Automated policy ingestion from email triggers.
4. Strict redaction feature for sensitive tokens.
5. Deploy embeddings/LLM using GPU nodes.

---

# 12. Implementation Roadmap

### Phase 1 — Foundation
- Setup Chroma, embeddings, LLM runtime.
- Build ingestion + vectorization pipeline.

### Phase 2 — Internal Chatbot
- Implement RBAC middleware.
- Add policy/IT document retrieval.
- Integrate ticket lookup.

### Phase 3 — Dashboard Development
- Document management UI.
- Versioning system.
- Department-level permissions.

### Phase 4 — Public Chatbot
- Deploy public-safe content pipeline.
- Implement website widget.

### Phase 5 — Hardening & Scaling
- Logging, analytics, monitoring.
- Add fallback APIs if required.

---

# 13. Conclusion

This system provides a secure, scalable, and efficient internal knowledge platform using a local RAG pipeline. It ensures sensitive information is appropriately protected while giving employees and external users fast, intelligent access to company knowledge.

