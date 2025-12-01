# 📋 Pending Tasks & Development Roadmap

## ✅ Recently Completed Features

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

### Advanced Features (Beyond Original Spec)
- ✅ **Document Versioning System** - Full version tracking with history
- ✅ **Google Gemini API Integration** - Production-ready cloud LLM option
- ✅ **Sentiment Analysis System** - Local sentiment classifier with tone detection
- ✅ **Guest User Onboarding System** - Progressive profile building
- ✅ **Support Chat Session Management** - Conversation history storage
- ✅ **Multi-Provider Query Architecture** - Unified interface across providers

## 🚀 Current Development Roadmap

### PHASE 1 — Stability, Accuracy & Core Intelligence

#### 1. Increase LLM Context + Token-Safe Pipeline ✅ COMPLETED
- Increase `n_ctx` to avoid overflow
- Add retry wrapper `_call_llm_with_retry()`

#### 2. Add Local Reranker 🔄 IN PROGRESS
- Use cross-encoder or BGE reranker
- Pipeline: Chroma → Top-K → Reranker → Top-N → LLM

#### 3. Add Prompt Caching 📋 PLANNED
- Cache embeddings, chunks, prefixes, and responses

#### 4. Add Explainability Endpoint `/rag/explain` 📋 PLANNED
- Return retrieved, filtered, selected chunks and scores

### PHASE 2 — Enterprise-Ready Assistants

#### 5. Intelligent Profile Routing 📋 PLANNED
- Based on role: HR, IT, Finance, Legal, Manager

#### 6. Onboarding Memory Expansion 📋 PLANNED
- Store preferences, writing style, past troubleshooting

#### 7. Query Reformulation 📋 PLANNED
- Rewrite queries to boost retrieval accuracy

### PHASE 3 — RAG Engine Upgrades

#### 8. Hybrid Retrieval 📋 PLANNED
- Combine BM25 + Vector + Reranker

#### 9. Document Versioning ✅ COMPLETED
- Track active/archived versions

#### 10. Chunk Summaries 📋 PLANNED
- Auto-summarize chunks during ingestion

### PHASE 4 — Security & Governance

#### 11. Redaction Mode 📋 PLANNED
- Replace sensitive parts with `[[REDACTED]]`

#### 12. Full Audit Logging 📋 PLANNED
- Log all retrievals, denied chunks, and decisions

### PHASE 5 — Analytics & Dashboards

#### 13. Usage Analytics Dashboard 📋 PLANNED
- Track top questions, docs, errors, latency

#### 14. Department Knowledge Health 📋 PLANNED
- Coverage reports for HR, IT, Finance, Legal

### PHASE 6 — Assistant Intelligence

#### 15. Autonomous Routing 📋 PLANNED
- Decide between RAG, LLM, follow-up question, or ticket

#### 16. Lightweight Reasoning Models 📋 PLANNED
- Use 2B–3B models for summarization, classification

#### 17. Ticketing Workflow 📋 PLANNED
- Auto create and summarize support tickets

### PHASE 7 — Offline Fine-Tuning / LoRA

#### 18. LoRA Fine-Tuning 📋 PLANNED
- Train adapters on internal policies and workflows

### PHASE 8 — Developer Experience & Scaling

#### 19. Background Ingestion Queue 📋 PLANNED
- Use FastAPI BackgroundTask or Celery

#### 20. Multi-Model Support 📋 PLANNED
- Switch between models dynamically

#### 21. Environment-Based Config Loader 📋 PLANNED
- Dev vs Production profiles

## ❌ Missing Features (High Priority)

### 1. Document Management APIs (Backend Only)
**Status:** Partially implemented  
**Missing Backend APIs:**
- ❌ Document approval workflow API (draft → approved → published)
- ❌ Enhanced document filtering (by department, status, version)

### 2. Ticket System Integration
**Status:** Not implemented  
**Missing Features:**
- ❌ Integration with ITSM/Jira Service Desk
- ❌ Ticket status queries ("What's the status of my ticket?")
- ❌ Show all open tickets for user
- ❌ Ticket database connection/API

### 3. Approval Workflow
**Status:** Not implemented  
**Missing Features:**
- ❌ Draft/pending status for documents
- ❌ Approval queue
- ❌ Approve/reject actions
- ❌ Publish after approval
- ❌ Role-based approval permissions

### 4. Public vs Internal Chatbot Separation
**Status:** Partially implemented  
**Missing Features:**
- ❌ Dedicated public chatbot endpoint/widget
- ❌ Dual content model (public summary + sensitive detail)
- ❌ Website widget integration
- ❌ Public content scrubbing/redaction

### 5. Audit Logging & Monitoring
**Status:** Basic logging only  
**Missing Features:**
- ❌ User interaction logs (who queried what, when)
- ❌ Document update audit trail
- ❌ Access attempt logs
- ❌ Failed permission logs
- ❌ Anonymized analytics
- ❌ Persistent log storage (DB/files)
- ❌ Audit log retrieval API

### 6. SSO Integration
**Status:** Not implemented  
**Missing Features:**
- ❌ Azure AD integration
- ❌ Google Workspace SSO
- ❌ LDAP authentication
- ❌ OAuth2 flows

## 🎯 TOP 10 IMMEDIATE NEXT STEPS

1. **Local Reranker** - Implement cross-encoder for better accuracy
2. **Prompt Caching** - Cache embeddings and responses
3. **`/rag/explain` Endpoint** - Add explainability for debugging
4. **Assistant Routing** - Role-based response routing
5. **Query Reformulation** - Improve retrieval with query rewriting
6. **Hybrid Retrieval** - Combine BM25 + Vector search
7. **Redaction Rules** - Implement content redaction
8. **Analytics Dashboard** - Usage and performance metrics
9. **Autonomous Routing Logic** - Smart decision making
10. **Ticket Integration** - Connect to ticketing systems

## 🔥 High-Impact, Low-Effort Improvements

### Core RAG Enhancements
- **Relevance Re-ranking** - Cross-encoder/MiniLM reranker after Chroma
- **Auto-summarization** - Summarize large chunks before storage
- **Hybrid retrieval** - TF-IDF + vector or BM25 + vector

### Security + RBAC Enhancements
- **Fine-grained access rules** - Classification level, region restrictions
- **Redaction before LLM** - Replace sensitive fields with [[REDACTED]]
- **Audit logging** - Enterprise-style activity logging

### Support Chat System Enhancements
- **Persistent long-term memory** - User preferences and history
- **Assistant personality modes** - HR, IT, Finance, Legal modes
- **Conversation continuation** - Resume sessions after restart
- **User sentiment detection** - Local BERT classifier for tone

### LLM Layer Upgrades
- **Model auto-selection** - Route tasks to appropriate model sizes
- **Offline fine-tuning** - LoRA training on company data
- **Local embedding model switch** - Support multiple embedding models
- **Prompt caching** - Store previous outputs and embeddings

### API / System Layer Enhancements
- **API Request Validation** - Reject malformed/oversized requests
- **Explainability endpoint** - `/rag/explain` for debugging
- **Background ingestion queue** - Handle large PDF ingestion
- **Configurable retrieval strategies** - User-selectable methods

## 📊 Evaluation & Testing Priorities

### Create Evaluation Suite
- Collect 100-300 representative Q&A pairs across departments
- Include easy, ambiguous, and restricted queries (RBAC)
- Add labeled "gold answer" or expected behavior
- Measure precision@k, MRR, exact-match, RBAC compliance

### Optimization Experiments
- **Chunking parameters** - Test chunk sizes (256, 512, 1024) and overlap
- **Token-budgeted selection** - Optimize chunk selection policy
- **Embeddings quality** - Compare different embedding models
- **RBAC stress tests** - Automated tests for policy violations
- **Prompt engineering** - A/B test different LLM prefixes
- **Factuality measures** - Reduce hallucination with source citing

## 📈 Success Metrics

### Retrieval Quality
- Precision@k, Recall@k, MRR
- End-to-end exact match / BLEU / ROUGE scores
- Hallucination rate (human or automatic checks)

### Security & Compliance
- False exposure rate (% of filtered docs returned)
- False denial rate (% of allowed docs blocked)
- RBAC policy violation count

### Performance
- Average response time (seconds)
- CPU usage per request
- User satisfaction rating (1-5 scale)

### Usage Analytics
- Most used documents
- Top errors and failure modes
- Most asked queries by department
- User engagement metrics

This roadmap provides a clear path from the current state to a production-ready enterprise RAG system with comprehensive features and robust security.