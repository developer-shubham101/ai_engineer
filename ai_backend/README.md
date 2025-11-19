# 🚀 **Local Role-Based Enterprise RAG System (CPU-Only, Offline)**

**AI-First Project Summary — For LLM Understanding**

This project is a **fully local, CPU-based enterprise RAG system** designed for **learning, experimentation, and offline testing**.
It does **not** depend on any external LLM APIs (OpenAI, Google, HuggingFace).
Everything runs locally using:

* **Mistral-7B-Instruct-v0.2.Q3_K_M.gguf**
* **Local embeddings (MiniLM)**
* **Local Chroma vector DB**
* **FastAPI backend**
* **Custom RBAC (Role-Based Access Control)**
* **Session-aware Support Chat System (SQLite)**

This README describes the project **from an AI / system-architecture perspective**, so any LLM can easily understand how the system works and give accurate help.

---

# 🧠 **1. High-Level Purpose**

This project simulates a **real enterprise AI assistant** inside a fictional company *Saarthi Infotech Pvt. Ltd.*

The system can:

### ✅ Answer role-specific questions

(Policies, workflows, HR, Finance, IT, Legal, etc.)

### ✅ Enforce strict RBAC

Five sensitivity levels:
`public_internal`, `department_confidential`, `role_confidential`, `highly_confidential`, `personal`

### ✅ Perform RAG queries locally

Using MiniLM embeddings + Chroma

### ✅ Run an LLM fully offline

Using Mistral-7B-Instruct GGUF via llama.cpp

### ✅ Maintain multi-turn support chat sessions

Stored in SQLite for short-term memory

### ✅ Provide natural-language AI responses

Built from allowed document chunks only

The entire system works **completely offline**, on a **CPU-only machine**, for **learning and testing**.

---

# 🏗 **2. System Architecture Overview**

```
User → FastAPI → RAG Pipeline → RBAC Filter → Local LLM → Response
                          ↓
                   Chroma Vector DB
                          ↓
                Local Embeddings (MiniLM)
```

### Components:

| Component                    | Purpose                                                     |
| ---------------------------- | ----------------------------------------------------------- |
| **FastAPI Server**           | Provides REST endpoints for query, add, seed, chat sessions |
| **RAG Local Service**        | Chunking, embeddings, Chroma querying, RBAC filtering       |
| **Local LLM (Mistral 7B)**   | Generates final natural-language answers                    |
| **ChromaDB**                 | Stores vector embeddings + metadata                         |
| **SQLite Support Chat**      | Memory + session history                                    |
| **Auth Layer**               | API-key based role simulation                               |
| **Role & Department System** | Controls document visibility                                |

---

# 🔐 **3. Role & Sensitivity Model**

### **Roles**

* Employee
* Manager
* HR
* Finance
* IT Support
* Legal
* Executive
* Guest

### **Sensitivity Levels**

* `public_internal` – visible to all
* `department_confidential` – only same department
* `role_confidential` – HR / Managers / Legal
* `highly_confidential` – Legal + Executives only
* `personal` – owner only (or HR/Legal/Exec)

This metadata is stored per document chunk and enforced after retrieval.

---

# 📚 **4. RAG Flow (AI-Focused Explanation)**

### 1. User asks a question

→ API receives: question, role, department, optional session.

### 2. Query text is embedded locally

Using **SentenceTransformers / MiniLM**.

### 3. Chroma returns top-k chunks

But these chunks may include restricted content.

### 4. **RBAC Filtering Happens**

Each chunk is checked by:

```
sensitivity
department
role
owner_id
allowed_roles
public_summary
```

Unauthorized chunks:

* Are removed
* But public summaries may be shown
* Count of filtered items is recorded

### 5. AI Prompt is built (optional session prefix)

Including:

* Support category (HR/IT/etc)
* Last 5 messages (history)
* User role / department context
* Allowed chunks only

### 6. Local Mistral LLM generates answer

Only using the allowed visible context.

---

# 💬 **5. Support Chat Subsystem**

A lightweight chat system stores:

* Session metadata
* User & assistant messages
* Timestamped history
* Conversation memory (last 5 turns)

This is used to build a dynamic prefix for the LLM.

### Example generated prompt prefix:

```
You are an HR support assistant.
User role: Employee, Department: Engineering
Conversation history:
[2025-01-10] USER: How do I apply for leave?
[2025-01-10] ASSISTANT: …
```

This allows local multi-turn AI behavior **without external APIs**.

---

# 🗂 **6. Document Ingestion**

### Two methods:

1. `/api/local/add` – JSON text
2. `/api/local/add-file` – Upload `.txt`

### Metadata includes:

```
department
sensitivity
allowed_roles
owner_id
public_summary
ingested_at
ingested_by
```

### Chunking:

* 512 chars
* 64 overlap
* Embedded locally

Chunks are then stored in ChromaDB.

---

# 🧪 **7. Local LLM (Mistral-7B-Instruct GGUF)**

The project uses:

**mistral-7b-instruct-v0.2.Q3_K_M.gguf**

Loaded through llama.cpp:

* CPU-only
* No internet
* Lazy loaded at first query
* Auto-detected from `/models/*.gguf`

---

# 🔌 **8. API Endpoints (Simplified)**

### **RAG**

* `POST /api/local/query`
* `POST /api/local/add`
* `POST /api/local/add-file`
* `POST /api/local/seed`

### **Support Chat**

* `POST /api/local/session/start`
* `POST /api/local/session/end`

### **LLM Helpers (local only)**

* `/summarize`
* `/generate`
* `/chat`
* `/ask-document`

---

# 🧩 **9. Project Goals**

This project is built for **AI engineers learning**:

* RAG pipelines
* RBAC security enforcement
* Local LLM inference
* Chat memory handling
* Metadata-aware document retrieval
* Enterprise knowledge systems
* End-to-end offline AI stack

It simulates what a **real company AI assistant** would look like.

---

# ⚙️ **10. What Makes This Project Unique**

* 100% offline
* CPU-only LLM
* Role & department sensitive content filtering
* Support chat with memory
* Zero cloud dependencies
* Realistic enterprise structure
* Clean separation of services
* Beginner-friendly but enterprise-grade concepts

---

# ✔️ **Summary**

This project is a **complete offline enterprise AI system**, combining:

* **Local LLM**
* **RAG**
* **RBAC**
* **Session memory**
* **Metadata-rich document ingestion**
* **FastAPI integration**

Designed specifically for **learning AI engineering**, not production.


=============================================================
perfect! my next task is that 

Task 1:

modify 
POST /api/local/session/start

in which it shoule simulate only oauth/okta (session creation) as user
hit it should create sesson in local/db (depending on current 
implimentation) and in response it should session id whic we'll 
pass in other ednpoint x-session-id, it should not ask anything from user make it empty body api. 
in case we need any modification in body we'll do it later on.

task 2:

after that modify for 
x-api-key: key-guest-1 (simulate role for now) 
POST /api/local/query

in which I want if base on session id
    
    senario 1
    if not and user added session key:
        then first question what evenr you ask let's say "hi/hello" 
        then like a support assistent ask what is your name -> Jon (user input in next response)
        then ask what is you gender -> Male (user input in next response) so on so far ask details of user
        if possibble add array for now to modify by developer
        after all query finish save these details in session related info only for that session 

    senario 2
    if exeist:
        when user asks any questions it ai should give answer like 
        Tell me about this company (user query)
        ai answer : sure, Jon (user name) this comapny is ....

        is there any opening for me I me in this company (user query)
            there is one requirement for for accountent (Details provided by user at senario 1 if availiable) base on your info 
            /
            there is 20 openings for you accountent/hr/developer... (Details we not provided by user in  senario 1 so ai don't know him)

    senario 3
    if not and user did not added session key:
        act as unknown user everything user ask if we need any info from user ask 



Note: we are feeding details to model add_document_to_rag_local so we need uses these data as well and other role functino should remain 
same

and you can change db for this requirment if is best to use croma the it's also fine if sqlite is fine then aslo good

i know cache Db like radis is best for that sinario but imagin if we want to store user/session info through out life time 

so user some after 10 days as wants to continue same chat
    

    





RAG

POST /api/local/query

POST /api/local/add

POST /api/local/add-file

POST /api/local/seed

Support Chat



POST /api/local/session/end








