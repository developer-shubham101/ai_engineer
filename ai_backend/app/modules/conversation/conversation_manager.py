"""Conversation history management — single messages table with chat_type discrimination."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import sqlite3
import uuid
import json
import logging

logger = logging.getLogger(__name__)

VALID_CHAT_TYPES = {"rag", "agent", "crew"}


class IConversationManager(ABC):

    @abstractmethod
    async def create_conversation(self, user_id: str, chat_type: str, title: Optional[str] = None) -> str:
        pass

    @abstractmethod
    async def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        pass

    @abstractmethod
    async def list_conversations(self, user_id: str, chat_type: Optional[str] = None,
                                  limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def update_conversation(self, conversation_id: str, user_id: str, **kwargs) -> bool:
        pass

    @abstractmethod
    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        pass

    @abstractmethod
    async def add_message(self, conversation_id: str, speaker: str, content: str,
                          chat_type: str, extra: Optional[Dict[str, Any]] = None) -> int:
        """Add a message to the unified messages table.
        extra: dict of chat_type-specific fields (serialised to JSON in extra_data column).
        """
        pass

    @abstractmethod
    async def get_messages(self, conversation_id: str, user_id: str,
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    async def generate_title(self, conversation_id: str) -> str:
        pass


class SQLiteConversationManager(IConversationManager):
    """SQLite implementation — single messages table with chat_type column."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._db_initialized = False
        self._init_db()

    def _init_db(self):
        if self._db_initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with sqlite3.connect(self.db_path) as conn:
            # conversations — now has chat_type
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT PRIMARY KEY,
                    user_id     TEXT NOT NULL,
                    chat_type   TEXT NOT NULL,
                    title       TEXT,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    is_archived BOOLEAN DEFAULT 0,
                    metadata    TEXT
                )
            """)

            # Single messages table — chat_type column + extra_data JSON for type-specific fields
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    chat_type       TEXT NOT NULL,
                    speaker         TEXT NOT NULL,
                    content         TEXT NOT NULL,
                    created_at      TEXT NOT NULL,

                    -- Common fields
                    processing_time_ms  INTEGER,
                    error_message       TEXT,

                    -- RAG-specific
                    user_query          TEXT,
                    llm_provider        TEXT,
                    llm_model           TEXT,
                    llm_prompt          TEXT,
                    llm_response_raw    TEXT,
                    llm_tokens_used     INTEGER,
                    llm_temperature     REAL,
                    llm_max_tokens      INTEGER,
                    retrieval_top_k     INTEGER,
                    use_documents       BOOLEAN,
                    use_llm             BOOLEAN,
                    sentiment           TEXT,
                    tone                TEXT,

                    -- Agent-specific
                    orchestrator_type   TEXT,

                    -- Crew-specific
                    workflow_type       TEXT,
                    iterations          INTEGER,

                    -- JSON blobs for complex fields
                    retrieved_context   TEXT,
                    embeddings_used     TEXT,
                    retrieved_doc_ids   TEXT,
                    sentiment_meta      TEXT,
                    tools_used          TEXT,
                    steps               TEXT,
                    agents_used         TEXT,

                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)

            conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_type ON conversations(chat_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_type ON messages(chat_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_msg_created ON messages(created_at DESC)")

            conn.commit()

        self._db_initialized = True
        logger.info("Conversations DB initialised at %s", self.db_path)

    def _connect(self) -> sqlite3.Connection:
        self._init_db()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── Conversation CRUD ─────────────────────────────────────────────────────

    async def create_conversation(self, user_id: str, chat_type: str,
                                   title: Optional[str] = None) -> str:
        if chat_type not in VALID_CHAT_TYPES:
            raise ValueError(f"Invalid chat_type '{chat_type}'. Valid: {sorted(VALID_CHAT_TYPES)}")

        conv_id = f"conv_{uuid.uuid4().hex}"
        ts = datetime.utcnow().isoformat() + "Z"

        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, user_id, chat_type, title, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (conv_id, user_id, chat_type, title or "New Conversation", ts, ts)
            )
            conn.commit()

        logger.info("CONV_CREATED: id=%s user=%s type=%s", conv_id, user_id, chat_type)
        return conv_id

    async def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, user_id, chat_type, title, created_at, updated_at, is_archived, metadata "
                "FROM conversations WHERE id = ? AND user_id = ? AND is_archived = 0",
                (conversation_id, user_id)
            ).fetchone()

            if not row:
                return None

            count = conn.execute(
                "SELECT COUNT(*) as c FROM messages WHERE conversation_id = ?",
                (conversation_id,)
            ).fetchone()

            return {**dict(row), "message_count": count["c"] if count else 0}

    async def list_conversations(self, user_id: str, chat_type: Optional[str] = None,
                                  limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            if chat_type:
                rows = conn.execute(
                    "SELECT c.id, c.user_id, c.chat_type, c.title, c.created_at, c.updated_at, "
                    "COUNT(m.id) as message_count "
                    "FROM conversations c LEFT JOIN messages m ON c.id = m.conversation_id "
                    "WHERE c.user_id = ? AND c.chat_type = ? AND c.is_archived = 0 "
                    "GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ? OFFSET ?",
                    (user_id, chat_type, limit, offset)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT c.id, c.user_id, c.chat_type, c.title, c.created_at, c.updated_at, "
                    "COUNT(m.id) as message_count "
                    "FROM conversations c LEFT JOIN messages m ON c.id = m.conversation_id "
                    "WHERE c.user_id = ? AND c.is_archived = 0 "
                    "GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ? OFFSET ?",
                    (user_id, limit, offset)
                ).fetchall()

            return [dict(r) for r in rows]

    async def update_conversation(self, conversation_id: str, user_id: str, **kwargs) -> bool:
        ts = datetime.utcnow().isoformat() + "Z"
        allowed = {"title", "metadata"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False

        updates["updated_at"] = ts
        set_clause = ", ".join(f"{k} = ?" for k in updates)
        values = list(updates.values()) + [conversation_id, user_id]

        with self._connect() as conn:
            cur = conn.execute(
                f"UPDATE conversations SET {set_clause} WHERE id = ? AND user_id = ?", values
            )
            conn.commit()
            return cur.rowcount > 0

    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        ts = datetime.utcnow().isoformat() + "Z"
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE conversations SET is_archived = 1, updated_at = ? WHERE id = ? AND user_id = ?",
                (ts, conversation_id, user_id)
            )
            conn.commit()
            return cur.rowcount > 0

    # ── Unified message write ─────────────────────────────────────────────────

    async def add_message(self, conversation_id: str, speaker: str, content: str,
                          chat_type: str, extra: Optional[Dict[str, Any]] = None) -> int:
        """Insert a message into the unified messages table.

        extra keys per chat_type:
          rag:   user_query, llm_provider, llm_model, llm_prompt, llm_response_raw,
                 llm_tokens_used, llm_temperature, llm_max_tokens, retrieval_top_k,
                 use_documents, use_llm, sentiment, tone,
                 retrieved_context (list), embeddings_used (dict),
                 retrieved_doc_ids (list), sentiment_meta (dict)
          agent: user_query, orchestrator_type,
                 tools_used (list), steps (list)
          crew:  user_query (topic), workflow_type, iterations,
                 agents_used (list)
        """
        if chat_type not in VALID_CHAT_TYPES:
            raise ValueError(f"Invalid chat_type '{chat_type}'")

        ts = datetime.utcnow().isoformat() + "Z"
        e = extra or {}

        # Serialise JSON blobs
        retrieved_context  = json.dumps(e.get("retrieved_context"))  if e.get("retrieved_context")  else None
        embeddings_used    = json.dumps(e.get("embeddings_used"))    if e.get("embeddings_used")    else None
        retrieved_doc_ids  = json.dumps(e.get("retrieved_doc_ids"))  if e.get("retrieved_doc_ids")  else None
        sentiment_meta     = json.dumps(e.get("sentiment_meta"))     if e.get("sentiment_meta")     else None
        tools_used         = json.dumps(e.get("tools_used"))         if e.get("tools_used")         else None
        steps              = json.dumps(e.get("steps"))              if e.get("steps")              else None
        agents_used        = json.dumps(e.get("agents_used"))        if e.get("agents_used")        else None

        with self._connect() as conn:
            cur = conn.execute("""
                INSERT INTO messages (
                    conversation_id, chat_type, speaker, content, created_at,
                    processing_time_ms, error_message,
                    user_query, llm_provider, llm_model, llm_prompt, llm_response_raw,
                    llm_tokens_used, llm_temperature, llm_max_tokens,
                    retrieval_top_k, use_documents, use_llm, sentiment, tone,
                    orchestrator_type,
                    workflow_type, iterations,
                    retrieved_context, embeddings_used, retrieved_doc_ids, sentiment_meta,
                    tools_used, steps, agents_used
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                conversation_id, chat_type, speaker, content, ts,
                e.get("processing_time_ms"), e.get("error_message"),
                e.get("user_query"), e.get("llm_provider"), e.get("llm_model"),
                e.get("llm_prompt"), e.get("llm_response_raw"),
                e.get("llm_tokens_used"), e.get("llm_temperature"), e.get("llm_max_tokens"),
                e.get("retrieval_top_k"), e.get("use_documents"), e.get("use_llm"),
                e.get("sentiment"), e.get("tone"),
                e.get("orchestrator_type"),
                e.get("workflow_type"), e.get("iterations"),
                retrieved_context, embeddings_used, retrieved_doc_ids, sentiment_meta,
                tools_used, steps, agents_used
            ))
            msg_id = cur.lastrowid

            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (ts, conversation_id))
            conn.commit()

        logger.debug("MSG_ADDED: id=%d conv=%s type=%s speaker=%s", msg_id, conversation_id, chat_type, speaker)
        return msg_id

    # ── Unified message read ──────────────────────────────────────────────────

    async def get_messages(self, conversation_id: str, user_id: str,
                           limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return messages for a conversation (ownership verified)."""
        conv = await self.get_conversation(conversation_id, user_id)
        if not conv:
            return []

        with self._connect() as conn:
            if limit:
                rows = conn.execute(
                    "SELECT * FROM (SELECT * FROM messages WHERE conversation_id = ? "
                    "ORDER BY created_at DESC LIMIT ?) ORDER BY created_at ASC",
                    (conversation_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                    (conversation_id,)
                ).fetchall()

        JSON_FIELDS = ["retrieved_context", "embeddings_used", "retrieved_doc_ids",
                       "sentiment_meta", "tools_used", "steps", "agents_used"]
        messages = []
        for row in rows:
            msg = dict(row)
            for key in JSON_FIELDS:
                if msg.get(key) and isinstance(msg[key], str):
                    try:
                        msg[key] = json.loads(msg[key])
                    except json.JSONDecodeError:
                        msg[key] = None
            messages.append(msg)

        return messages

    async def generate_title(self, conversation_id: str) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT content FROM messages WHERE conversation_id = ? AND speaker = 'user' "
                "ORDER BY created_at ASC LIMIT 1",
                (conversation_id,)
            ).fetchone()
            if row:
                c = row["content"]
                return c[:50] + "..." if len(c) > 50 else c
            return "New Conversation"

    # ── Legacy compatibility helpers (used by existing /query routes) ─────────

    async def add_rag_message(self, conversation_id: str, speaker: str, content: str,
                               **kwargs) -> int:
        return await self.add_message(conversation_id, speaker, content, "rag", extra=kwargs)

    async def add_agent_message(self, conversation_id: str, speaker: str, content: str,
                                 **kwargs) -> int:
        return await self.add_message(conversation_id, speaker, content, "agent", extra=kwargs)

    async def add_crew_message(self, conversation_id: str, speaker: str, content: str,
                                **kwargs) -> int:
        return await self.add_message(conversation_id, speaker, content, "crew", extra=kwargs)
