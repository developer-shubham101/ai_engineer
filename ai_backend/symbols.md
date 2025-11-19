Support-focused local RAG backend for Saarthi Infotech: ChromaDB + MiniLM embeddings + LlamaCpp, FastAPI routes, plus new SQLite-backed session-aware chat layer. RBAC metadata gates document access and logs public summaries for blocked chunks. Repo includes comprehensive instruction docs describing structure, goals, and pending tasks. SQLite sessions reset on startup; `/api/local/query` now honors `X-Session-Id` for multi-turn assistance, while `/api/local/session/*` handles lifecycle. Models/logs/chroma storage are large binary areas to skip. Work left: compliance metrics, RBAC matrix doc, broader support-ticket features.

{
  "project_summary": {
    "name": "ai_backend",
    "purpose": "Local-only enterprise RAG assistant with role-based access control and support-chat workflows for Saarthi Infotech.",
    "stack": [
      "Python 3",
      "FastAPI",
      "ChromaDB",
      "sentence-transformers MiniLM",
      "LlamaCpp",
      "SQLite"
    ],
    "progress_percent": "≈70%"
  },
  "high_priority_files": [
    "instruction.txt",
    "docs/requirment.md",
    "docs/Role-based enterprise AI design.md",
    "app/services/rag_local_service.py",
    "app/api_routes_local.py",
    "app/services/support_chat.py"
  ],
  "file_map": [
    {
      "file": "instruction.txt",
      "symbols": [
        {
          "symbol_name": "(module-level)",
          "kind": "module",
          "brief_description": "Living project brief: tool guidance, requirements snapshot, implementation status, API summaries, pending tasks.",
          "inputs_outputs": "Inputs: human readers; Output: context for future contributors.",
          "sensitivity_tags": []
        }
      ],
      "notes": "Read first; tracks progress log and instructions for Cursor/Copilot."
    },
    {
      "file": "docs/requirment.md",
      "symbols": [
        {
          "symbol_name": "(module-level)",
          "kind": "module",
          "brief_description": "Detailed requirements for the CPU-only RAG+RBAC assistant, including tone, key decisions, and support-chat roadmap.",
          "inputs_outputs": "Input: project planning; Output: reference doc.",
          "sensitivity_tags": [
            "public_internal"
          ]
        }
      ]
    },
    {
      "file": "docs/Role-based enterprise AI design.md",
      "symbols": [
        {
          "symbol_name": "(module-level)",
          "kind": "module",
          "brief_description": "Defines company roles, information categories, sensitivity tiers, and behavioral examples for the AI assistant.",
          "inputs_outputs": "Input: organizational design; Output: policy reference.",
          "sensitivity_tags": [
            "public_internal"
          ]
        }
      ]
    },
    {
      "file": "app/services/rag_local_service.py",
      "symbols": [
        {
          "symbol_name": "initialize_local_rag",
          "kind": "function",
          "brief_description": "Prepares embedding/LLM instances and ensures the Chroma collection exists.",
          "inputs_outputs": "Inputs: optional embedding/LLM instances, persist dir, collection name; Output: None (side effects).",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "add_document_to_rag_local",
          "kind": "function",
          "brief_description": "Chunks text, sanitizes metadata, computes embeddings, and adds documents to Chroma with RBAC tags.",
          "inputs_outputs": "Inputs: source_name, text/chunks, metadata; Output: list of chunk IDs.",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "query_local_rag",
          "kind": "function",
          "brief_description": "Embeds query, retrieves top-k chunks, applies RBAC filtering, aggregates context, and optionally invokes LlamaCpp.",
          "inputs_outputs": "Inputs: query text, n_results, requester dict, prompt prefix, use_llm flag, max tokens; Output: dict containing visible/raw docs, context, summaries.",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "_allowed_by_metadata",
          "kind": "function",
          "brief_description": "Internal RBAC rule evaluator for personal, department, role, and highly confidential sensitivities.",
          "inputs_outputs": "Inputs: metadata dict, requester dict; Output: boolean allow flag.",
          "sensitivity_tags": [
            "role_confidential"
          ]
        },
        {
          "symbol_name": "seed_from_file",
          "kind": "function",
          "brief_description": "Loads default data files or directories into the RAG index for demo/testing.",
          "inputs_outputs": "Inputs: file_path, source_name; Output: list of chunk IDs added.",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "update_metadata",
          "kind": "function",
          "brief_description": "Updates metadata for existing chunk IDs in Chroma.",
          "inputs_outputs": "Inputs: ids list, metadata dict; Output: bool success.",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "clear_collection",
          "kind": "function",
          "brief_description": "Deletes all documents from the configured Chroma collection.",
          "inputs_outputs": "Inputs: none; Output: None (side effects).",
          "sensitivity_tags": []
        }
      ]
    },
    {
      "file": "app/services/support_chat.py",
      "symbols": [
        {
          "symbol_name": "init_support_chat_db",
          "kind": "function",
          "brief_description": "Initializes/reset the SQLite database storing chat sessions and messages.",
          "inputs_outputs": "Inputs: reset flag; Output: None (creates tables).",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "create_session",
          "kind": "function",
          "brief_description": "Creates a new session row with role/department and timestamps, returning the session ID.",
          "inputs_outputs": "Inputs: optional session_id, role, department; Output: session ID string.",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "touch_session",
          "kind": "function",
          "brief_description": "Updates session role/department and last-updated timestamp.",
          "inputs_outputs": "Inputs: session ID, role, department; Output: None.",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "end_session",
          "kind": "function",
          "brief_description": "Deletes a session and its messages.",
          "inputs_outputs": "Inputs: session ID; Output: None.",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "store_message",
          "kind": "function",
          "brief_description": "Appends a message from user/assistant/system into the SQLite log.",
          "inputs_outputs": "Inputs: session ID, speaker, content; Output: None.",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "fetch_recent_messages",
          "kind": "function",
          "brief_description": "Retrieves the most recent N messages for a session in chronological order.",
          "inputs_outputs": "Inputs: session ID, limit; Output: list of dicts.",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "build_prompt_prefix",
          "kind": "function",
          "brief_description": "Constructs persona/context prompt text using requester role, department, category, and chat history.",
          "inputs_outputs": "Inputs: requester dict, history list, category string; Output: prompt prefix string.",
          "sensitivity_tags": []
        }
      ]
    },
    {
      "file": "app/api_routes_local.py",
      "symbols": [
        {
          "symbol_name": "POST /api/local/query",
          "kind": "function",
          "brief_description": "Handles questions against the local RAG; optionally uses `X-Session-Id` to load chat history and log new turns.",
          "inputs_outputs": "Inputs: QueryRequest (question/top_k/use_llm/max_tokens/category), headers (`X-API-Key`, `X-Session-Id`); Output: QueryResponse with answer, retrieved docs, context.",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "POST /api/local/add",
          "kind": "function",
          "brief_description": "Ingests inline text with metadata tags for RBAC.",
          "inputs_outputs": "Inputs: AddDocRequest; Output: AddResponse (message, chunk_count).",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "POST /api/local/add-file",
          "kind": "function",
          "brief_description": "Uploads text files (≤5 MB) for ingestion with department+sensitivity metadata.",
          "inputs_outputs": "Inputs: UploadFile, query params; Output: AddResponse.",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "POST /api/local/seed",
          "kind": "function",
          "brief_description": "Seeds default company data into the RAG index.",
          "inputs_outputs": "Inputs: none; Output: AddResponse.",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "POST /api/local/clear",
          "kind": "function",
          "brief_description": "Clears the Chroma collection (restricted to Executive/Legal).",
          "inputs_outputs": "Inputs: none; Output: AddResponse.",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "POST /api/local/session/start",
          "kind": "function",
          "brief_description": "Creates a new support chat session and logs optional profile info as a system message.",
          "inputs_outputs": "Inputs: SupportSessionStartRequest; Output: SupportSessionStartResponse.",
          "sensitivity_tags": []
        },
        {
          "symbol_name": "POST /api/local/session/end",
          "kind": "function",
          "brief_description": "Ends a chat session and deletes associated history.",
          "inputs_outputs": "Inputs: SupportSessionEndRequest; Output: SupportSessionEndResponse.",
          "sensitivity_tags": []
        }
      ]
    },
    {
      "file": "app/services/auth.py",
      "symbols": [
        {
          "symbol_name": "get_user_from_api_key",
          "kind": "function",
          "brief_description": "Simple API-key lookup returning user_id, role, department, including guest/demo keys.",
          "inputs_outputs": "Inputs: API key string; Output: user dict or None.",
          "sensitivity_tags": [
            "public_internal"
          ]
        }
      ],
      "notes": "Contains demo API keys; update for production RBAC."
    },
    {
      "file": "docs/task_for_ai.md",
      "symbols": [
        {
          "symbol_name": "(module-level)",
          "kind": "module",
          "brief_description": "Task backlog for AI assistants; includes role-based scenarios and checklist for future work.",
          "inputs_outputs": "Documentation only.",
          "sensitivity_tags": []
        }
      ]
    },
    {
      "file": "docs/download_model.md",
      "symbols": [
        {
          "symbol_name": "(module-level)",
          "kind": "module",
          "brief_description": "Instructions for obtaining local GGUF models for LlamaCpp.",
          "inputs_outputs": "Documentation only.",
          "sensitivity_tags": []
        }
      ]
    },
    {
      "file": "docs/alternative_lib.md",
      "symbols": [
        {
          "symbol_name": "(module-level)",
          "kind": "module",
          "brief_description": "Lists optional libraries/approaches for embeddings, inference, and storage.",
          "inputs_outputs": "Documentation only.",
          "sensitivity_tags": []
        }
      ]
    },
    {
      "file": "scripts/seed_examples.py",
      "symbols": [
        {
          "symbol_name": "(module-level)",
          "kind": "script",
          "brief_description": "Command-line helper to ingest example HR/Finance documents with metadata for RBAC testing.",
          "inputs_outputs": "Inputs: script args; Output: seeds data into Chroma.",
          "sensitivity_tags": []
        }
      ]
    },
    {
      "file": "scripts/local_tree_printer.py",
      "symbols": [
        {
          "symbol_name": "(module-level)",
          "kind": "script",
          "brief_description": "Utility to print the repo tree while skipping heavy folders.",
          "inputs_outputs": "Inputs: filesystem path; Output: console tree.",
          "sensitivity_tags": []
        }
      ],
      "notes": "Optional tooling, not core logic."
    }
  ],
  "skip_list": [
    {
      "path": "app/chroma_storage/",
      "reason": "Vector database persists embeddings; large binary files, do not scan."
    },
    {
      "path": "models/",
      "reason": "Local GGUF/LLM binaries; heavy, no source."
    },
    {
      "path": "logs/",
      "reason": "Runtime logs only."
    },
    {
      "path": "app/data/sessions/",
      "reason": "Runtime session/save data, not code."
    },
    {
      "path": "data/missions_output/",
      "reason": "Large example corpora; not core logic."
    },
    {
      "path": ".idea/, .cache/",
      "reason": "Editor and cache metadata."
    }
  ],
  "next_steps": [
    "Review instruction.txt to understand tooling guidance, recent progress, and pending tasks.",
    "Study docs/requirment.md and docs/Role-based enterprise AI design.md for RBAC/business context.",
    "Inspect app/services/rag_local_service.py to learn the RAG + RBAC pipeline.",
    "Examine app/api_routes_local.py to see FastAPI endpoints, especially session-aware /api/local/query.",
    "Review app/services/support_chat.py to understand SQLite session handling and prompt building."
  ]
}

Action Checklist:
- Sync local environment, ensure embeddings/model files exist.
- Rerun tests or manual flow: POST /session/start → /query (with X-Session-Id) → /session/end.
- Plan compliance metrics/RBAC matrix documentation.
- Build forthcoming support-ticket, escalation, and audit features.