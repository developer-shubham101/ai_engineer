# 🚀 Multi-Provider Enterprise RAG System

## Project Motivation: Learning RAG Systems & Tools

This repository is designed as a **comprehensive reference implementation** for AI Engineers looking to master Retrieval-Augmented Generation (RAG) systems. It serves as a practical, hands-on playground to understand the complexities of building production-grade AI applications, moving beyond simple demos to handle real-world challenges like:

*   **Multi-Provider Orchestration**: How to switch seamlessly between OpenAI, Google, Hugging Face, and Local LLMs.
*   **Enterprise Constraints**: Implementing RBAC (Role-Based Access Control), document versioning, and audit logging.
*   **System Architecture**: connecting a FastAPI backend with a modern React frontend.

## Project Overview

While built with the robustness of a production system, the primary goal of this codebase is **education**. It demonstrates a robust, modular architecture for a Multi-Provider Enterprise RAG system. It supports both offline-first (local models) and cloud-based (API) LLM providers through a unified interface. Designed for enterprise environments, it features comprehensive Role-Based Access Control (RBAC), document versioning, and intelligent session management. A lightweight React frontend is included to demonstrate how to build user interfaces that interact effectively with complex RAG backends.

---

## ✨ Features

### Backend
*   **Multi-Provider LLM Support**: Seamlessly integrate with local models (Mistral-7B, Phi-2, Llama-3.2, Gemma-2B via llama-cpp-python) and cloud APIs (Google Gemini-2.5-Flash/Pro, OpenAI GPT-3.5/4, Hugging Face Inference API).
*   **Enterprise RBAC**: Flexible role overrides, department restrictions, and level-based validation for granular access control to documents and functionalities.
*   **Document Versioning**: Non-destructive updates, allowing for tracking of changes and comparison between document versions.
*   **Session-Aware Conversations**: Persistent, isolated conversation history with user profile integration for personalized and tone-aware responses.
*   **Prompt Optimization**: Dynamic prompt construction with token budgeting, context truncation, and a Chain of Responsibility pattern for flexible prompt building.
*   **Local Model Training**: Fine-tune Llama 3.2 1B on company data, export to HuggingFace and GGUF formats, and integrate seamlessly into the RAG system.
*   **Comprehensive API**: RESTful API endpoints for RAG queries, document management (CRUD), authentication, and model management.
*   **Enhanced Logging & Debugging**: Performance metrics, RBAC audit trails, and LLM interaction logs for comprehensive monitoring.
*   **Unified Temperature Control**: A single `temperature` parameter (0.0-1.0) controls response creativity across all integrated LLM providers.

### Frontend (React)
*   **Interactive Chat Interface**: Engage with the RAG system using a user-friendly chat UI.
*   **Document Management UI**: Upload files, ingest JSON documents, and manage document versions through intuitive forms.
*   **Admin Panels**: Tools for access-requests, metadata updates, and testing RBAC functionalities.
*   **Authentication**: JWT token-based authentication with user role determination for personalized experiences.
*   **Local LLM Model Selection**: Directly select specific local models for queries via the UI.

---

## 🏗️ Architecture Overview

The system follows a **modular architecture** with a FastAPI backend and a React frontend.

### Backend (`ai_backend/`)
Built with FastAPI, the backend is structured into distinct modules for Authentication, LLM, Vector DB, Core Business Logic, Configuration, and API layer components. It utilizes ChromaDB for vector storage, SQLite for session/user/version management, and integrates various LLMs via a unified `BaseRAGService`. Dependency Injection is managed through a dedicated container for testability and maintainability.

### Frontend (`front_end_react/`)
A lightweight React application bootstrapped with Vite, providing the user interface for interacting with the backend. It handles authentication, RAG queries, document uploads, and administrative tasks, leveraging the backend's REST APIs.

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   Node.js (for frontend)
*   `pip` and `npm` (or `yarn`)

### Backend Setup
1.  Navigate to the `ai_backend` directory:
    ```bash
    cd ai_backend
    ```
2.  Install backend dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download embedding models and optionally local LLMs:
    ```bash
    python scripts/download_embeddings_models.py
    # To download all local LLMs (optional, can be large):
    python scripts/download_hf_model.py --all
    ```
4.  Run the FastAPI application:
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    # Or for production:
    # uvicorn app.main:app --host 0.0.0.0 --port 8000
    ```
    The backend will be accessible at `http://localhost:8000`.

### Frontend Setup
1.  Navigate to the `front_end_react` directory:
    ```bash
    cd front_end_react
    ```
2.  Install frontend dependencies:
    ```bash
    npm install
    # or yarn install
    ```
3.  Start the React development server:
    ```bash
    npm run dev
    # or yarn dev
    ```
    The frontend will typically be accessible at `http://localhost:5173`. Ensure `BASE_API_URL` in `src/utility/const.js` points to your backend (default `http://localhost:8000`).

---

## 📖 API Documentation

For detailed information on all available API endpoints, request/response formats, authentication, and examples, please refer to:

[API_DOCUMENTATION.md](ai_backend/documents/API_DOCUMENTATION.md)

---

## 🔒 RBAC and Security

The system implements a comprehensive Role-Based Access Control (RBAC) system with a role hierarchy (`Guest` < `Employee` < `HR` < `Manager` < `SuperAdmin`) and sensitivity levels for documents (`public_internal`, `department_confidential`, `role_confidential`, `highly_confidential`, `personal`). Access is enforced at the document level, ensuring data security and compliance. JWT authentication secures all API interactions.

---

## 🤖 Model Management

The system supports dynamic management of LLMs, allowing configuration of local GGUF models (auto-detected from the `models/` directory) and integration with various cloud providers. You can list available models, download new ones, and fine-tune custom Llama models on your enterprise data.

---

## 📄 Document Management

Documents are managed with robust versioning capabilities. Each update creates a new, non-destructive version, allowing for historical tracking and comparison. Metadata attached to documents (department, sensitivity, owner, allowed roles) is used by the RBAC system to control access.

---

## 🤝 Contributing

Contributions are welcome! Please fork the repository, create a new branch for your features or bug fixes, and submit a pull request.

---

## 🧑‍💻 Author

**Shubham Sharma**
[GitHub Profile](https://github.com/developer-shubham101)