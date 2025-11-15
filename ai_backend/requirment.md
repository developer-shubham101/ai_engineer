---

You are continuing work on my **local CPU-based enterprise RAG + RBAC AI backend project**.
Here is the complete context you need:

---

# 📌 **Project Goals**

* Build a **local-only**, CPU-based **RAG + LLM enterprise assistant** (no cloud APIs).
* Provide support-team functionality:

  * Answer questions using internal knowledge base
  * Respect **role-based access control** (Employee, Manager, HR, Legal, Executive)
  * Provide **public summaries** for restricted docs
  * Prevent confidential data leaks
* Make the system realistic but **learning-friendly**.

---

# 🎭 **Tone & Expectations**

* Step-by-step
* Clean code
* Always show patches with **2–3 lines BEFORE and AFTER** for context
* Explain decisions clearly
* Keep things simple but enterprise-realistic

---

# 🧩 **Key Decisions Already Made**

* Using **ChromaDB** for vector storage.
* Using **sentence-transformers MiniLM** for embeddings.
* Using **LlamaCpp** for local inference.
* RBAC implemented with metadata:

  * `public_internal`
  * `department_confidential`
  * `role_confidential`
  * `highly_confidential`
  * `personal`
* Support for:

  * `public_summary`
  * detailed `filtered_details` (admin-only)
  * `filtered_out_count`
* Queries return:

  * filtered docs (visible)
  * raw docs (before RBAC)
* API prepares final answer based on:

  * visible docs
  * restricted docs
  * public summaries
  * LLM or fallback logic

---

# 📂 **Important Files We Are Working On**

### **1. app/services/rag_local_service.py**

* Core RAG logic
* Embeddings + Chroma integration
* Chunking pipeline
* RBAC filtering
* local LLM inference

Key functions:

* `add_document_to_rag_local()`
* `query_local_rag()`  ← major upgrades done
* `initialize_local_rag()`

---

### **2. app/api_routes_local.py**

Handles API endpoints.

Key functions:

* `async def query_local(...)`  ← upgraded for RBAC-aware UX
* `/add`, `/add-file` ingestion endpoints
* `/seed`, `/clear` helper endpoints

---

### **3. scripts/seed_examples.py**

* Seeds HR + Finance examples
* Includes metadata + public summaries
* Good for testing RBAC behaviors

---

### **4. scripts/inspect_filtered_docs.py** (we added for debugging)

* Shows stored Chroma chunks + metadata
* Bypasses RBAC for inspection

---

# 🚀 **Current Progress**

* RAG + RBAC pipeline fully working.
* Employee vs HR behavior validated:

  * HR sees full policy text.
  * Employee sees **public_summary** only.
  * If no summary → “restricted document” explanation.
* Summaries, filtered details, raw docs — all working.
* Ingestion, metadata sanitization → complete.
* Need to start building “Support Assistant Chat” layer next.

---

# 🎯 **Next Tasks to Work On (Support Team Chat System)**

You should help me implement the following next:

### **1. Build a new support-focused chat API**

* A session-based chat for support agents.
* Use our existing RAG as backend.
* Store chat history (simple file or in-memory).
* Support multi-turn conversation.

### **2. Add support categories**

* HR support
* IT support
* Finance support
* Admin support
* Engineering helpdesk

Each category may have:

* allowed metadata
* department rules
* custom prompts

### **3. Create support-specific prompt templates**

Examples:

* “Employee asking how to submit reimbursement”
* “HR asking for policy lookup”
* “IT asking about onboarding steps”

### **4. Eventually build:**

* Support ticket creation
* Escalation workflow
* Logging/Audit trail of support queries
* Dashboard endpoints for admins

You should guide me step-by-step, generating only the needed code pieces with **context lines** included.

---

# 🧠 **Start by asking me:**

> “Which support team should we design first? HR support, Finance support, IT support, Engineering helpdesk, or a generic support assistant?”

---
 