"""Conversation history management with comprehensive RAG logging."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from datetime import datetime
from pathlib import Path
import sqlite3
import uuid
import json
import logging

logger = logging.getLogger(__name__)


class IConversationManager(ABC):
    """Interface for conversation history management."""
    
    @abstractmethod
    async def create_conversation(self, user_id: str, title: Optional[str] = None) -> str:
        """Create a new conversation for a user."""
        pass
    
    @abstractmethod
    async def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific conversation (with ownership check)."""
        pass
    
    @abstractmethod
    async def list_conversations(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List all conversations for a user."""
        pass
    
    @abstractmethod
    async def update_conversation(self, conversation_id: str, user_id: str, **kwargs) -> bool:
        """Update conversation metadata (title, etc.)."""
        pass
    
    @abstractmethod
    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation (soft delete)."""
        pass
    
    @abstractmethod
    async def add_crew_message(
        self,
        conversation_id: str,
        speaker: str,
        content: str,
        user_topic: Optional[str] = None,
        workflow_type: Optional[str] = None,
        agents_used: Optional[List[str]] = None,
        iterations: Optional[int] = None,
        processing_time_ms: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> int:
        """Add a message with CrewAI workflow logging to crew_messages table."""
        pass

    @abstractmethod
    async def add_agent_message(
        self,
        conversation_id: str,
        speaker: str,
        content: str,
        # Agent-specific fields
        user_query: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        orchestrator_type: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> int:
        """Add a message with agent pipeline logging (steps, tools used)."""
        pass

    @abstractmethod
    async def add_message(self, conversation_id: str, speaker: str, content: str, 
                         sentiment: Optional[str] = None, tone: Optional[str] = None) -> int:
        """Add a simple message to a conversation (for basic user/assistant messages)."""
        pass
    
    @abstractmethod
    async def add_rag_message(
        self,
        conversation_id: str,
        speaker: str,
        content: str,
        # RAG Pipeline Data
        user_query: Optional[str] = None,
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
        embeddings_used: Optional[Dict[str, Any]] = None,
        llm_prompt: Optional[str] = None,
        llm_response_raw: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_tokens_used: Optional[int] = None,
        llm_temperature: Optional[float] = None,
        llm_max_tokens: Optional[int] = None,
        retrieved_doc_ids: Optional[List[str]] = None,
        retrieval_top_k: Optional[int] = None,
        use_documents: bool = True,
        use_llm: bool = True,
        processing_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        # Sentiment/Tone
        sentiment: Optional[str] = None,
        tone: Optional[str] = None,
        sentiment_meta: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Add a message with full RAG pipeline logging.
        This stores comprehensive information about the RAG query execution.
        """
        pass
    
    @abstractmethod
    async def get_messages(self, conversation_id: str, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages from a conversation."""
        pass
    
    @abstractmethod
    async def generate_title(self, conversation_id: str) -> str:
        """Auto-generate a conversation title from first messages."""
        pass


class SQLiteConversationManager(IConversationManager):
    """SQLite implementation of conversation manager with RAG logging."""
    
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._db_initialized = False
        self._init_db()
    
    def _init_db(self):
        """Initialize the conversations database with comprehensive RAG logging schema."""
        if self._db_initialized:
            return
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            # Conversations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    is_archived BOOLEAN DEFAULT 0,
                    metadata TEXT
                )
            """)
            
            # Messages table with comprehensive RAG logging
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    
                    -- Sentiment/Tone
                    sentiment TEXT,
                    tone TEXT,
                    sentiment_meta TEXT,
                    
                    -- RAG Pipeline Logging
                    user_query TEXT,
                    retrieved_context TEXT,
                    embeddings_used TEXT,
                    llm_prompt TEXT,
                    llm_response_raw TEXT,
                    llm_provider TEXT,
                    llm_model TEXT,
                    llm_tokens_used INTEGER,
                    llm_temperature REAL,
                    llm_max_tokens INTEGER,
                    retrieved_doc_ids TEXT,
                    retrieval_top_k INTEGER,
                    use_documents BOOLEAN DEFAULT 1,
                    use_llm BOOLEAN DEFAULT 1,
                    processing_time_ms INTEGER,
                    error_message TEXT,
                    
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            
            # Indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC)")

            # Agent conversations table — separate from RAG messages
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    user_query TEXT,
                    tools_used TEXT,
                    steps TEXT,
                    orchestrator_type TEXT,
                    processing_time_ms INTEGER,
                    error_message TEXT,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_messages_conv_id ON agent_messages(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_messages_created_at ON agent_messages(created_at DESC)")

            # Crew messages table — separate from RAG and agent messages
            conn.execute("""
                CREATE TABLE IF NOT EXISTS crew_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    user_topic TEXT,
                    workflow_type TEXT,
                    agents_used TEXT,
                    iterations INTEGER,
                    processing_time_ms INTEGER,
                    error_message TEXT,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_crew_messages_conv_id ON crew_messages(conversation_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_crew_messages_created_at ON crew_messages(created_at DESC)")
            
            conn.commit()
        
        self._db_initialized = True
        logger.info(f"Initialized conversations database at {self.db_path}")
    
    def _connect(self) -> sqlite3.Connection:
        """Internal connection helper."""
        self._init_db()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    async def create_conversation(self, user_id: str, title: Optional[str] = None) -> str:
        """Create a new conversation for a user."""
        conv_id = f"conv_{uuid.uuid4().hex}"
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO conversations (id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (conv_id, user_id, title or "New Conversation", timestamp, timestamp))
            conn.commit()
        
        logger.info(f"Created conversation {conv_id} for user {user_id}")
        return conv_id
    
    async def get_conversation(self, conversation_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific conversation (with ownership check)."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT id, user_id, title, created_at, updated_at, is_archived, metadata
                FROM conversations
                WHERE id = ? AND user_id = ? AND is_archived = 0
            """, (conversation_id, user_id)).fetchone()
            
            if not row:
                return None
            
            # Get message count
            count_row = conn.execute("""
                SELECT COUNT(*) as count FROM messages WHERE conversation_id = ?
            """, (conversation_id,)).fetchone()
            
            return {
                **dict(row),
                "message_count": count_row["count"] if count_row else 0
            }
    
    async def list_conversations(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List all conversations for a user."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT c.id, c.user_id, c.title, c.created_at, c.updated_at, c.is_archived,
                       COUNT(m.id) as message_count
                FROM conversations c
                LEFT JOIN messages m ON c.id = m.conversation_id
                WHERE c.user_id = ? AND c.is_archived = 0
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                LIMIT ? OFFSET ?
            """, (user_id, limit, offset)).fetchall()
            
            return [dict(row) for row in rows]
    
    async def update_conversation(self, conversation_id: str, user_id: str, **kwargs) -> bool:
        """Update conversation metadata (title, etc.)."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Build update query dynamically
        allowed_fields = {"title", "metadata"}
        updates = {k: v for k, v in kwargs.items() if k in allowed_fields}
        
        if not updates:
            return False
        
        updates["updated_at"] = timestamp
        
        set_clause = ", ".join([f"{k} = ?" for k in updates.keys()])
        values = list(updates.values()) + [conversation_id, user_id]
        
        with self._connect() as conn:
            cursor = conn.execute(f"""
                UPDATE conversations
                SET {set_clause}
                WHERE id = ? AND user_id = ?
            """, values)
            conn.commit()
            
            return cursor.rowcount > 0
    
    async def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        """Delete a conversation (soft delete)."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        with self._connect() as conn:
            cursor = conn.execute("""
                UPDATE conversations
                SET is_archived = 1, updated_at = ?
                WHERE id = ? AND user_id = ?
            """, (timestamp, conversation_id, user_id))
            conn.commit()
            
            return cursor.rowcount > 0
    
    async def add_message(self, conversation_id: str, speaker: str, content: str, 
                         sentiment: Optional[str] = None, tone: Optional[str] = None) -> int:
        """Add a simple message to a conversation (for basic user/assistant messages)."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO messages (conversation_id, speaker, content, created_at, sentiment, tone)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (conversation_id, speaker, content, timestamp, sentiment, tone))
            message_id = cursor.lastrowid
            
            # Update conversation timestamp
            conn.execute("""
                UPDATE conversations SET updated_at = ? WHERE id = ?
            """, (timestamp, conversation_id))
            
            conn.commit()
            
        return message_id
    
    async def add_rag_message(
        self,
        conversation_id: str,
        speaker: str,
        content: str,
        # RAG Pipeline Data
        user_query: Optional[str] = None,
        retrieved_context: Optional[List[Dict[str, Any]]] = None,
        embeddings_used: Optional[Dict[str, Any]] = None,
        llm_prompt: Optional[str] = None,
        llm_response_raw: Optional[str] = None,
        llm_provider: Optional[str] = None,
        llm_model: Optional[str] = None,
        llm_tokens_used: Optional[int] = None,
        llm_temperature: Optional[float] = None,
        llm_max_tokens: Optional[int] = None,
        retrieved_doc_ids: Optional[List[str]] = None,
        retrieval_top_k: Optional[int] = None,
        use_documents: bool = True,
        use_llm: bool = True,
        processing_time_ms: Optional[int] = None,
        error_message: Optional[str] = None,
        # Sentiment/Tone
        sentiment: Optional[str] = None,
        tone: Optional[str] = None,
        sentiment_meta: Optional[Dict[str, Any]] = None
    ) -> int:
        """Add a message with full RAG pipeline logging."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        
        # Serialize complex data to JSON
        retrieved_context_json = json.dumps(retrieved_context) if retrieved_context else None
        embeddings_used_json = json.dumps(embeddings_used) if embeddings_used else None
        sentiment_meta_json = json.dumps(sentiment_meta) if sentiment_meta else None
        retrieved_doc_ids_str = ",".join(retrieved_doc_ids) if retrieved_doc_ids else None
        
        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO messages (
                    conversation_id, speaker, content, created_at,
                    sentiment, tone, sentiment_meta,
                    user_query, retrieved_context, embeddings_used,
                    llm_prompt, llm_response_raw, llm_provider, llm_model,
                    llm_tokens_used, llm_temperature, llm_max_tokens,
                    retrieved_doc_ids, retrieval_top_k,
                    use_documents, use_llm, processing_time_ms, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation_id, speaker, content, timestamp,
                sentiment, tone, sentiment_meta_json,
                user_query, retrieved_context_json, embeddings_used_json,
                llm_prompt, llm_response_raw, llm_provider, llm_model,
                llm_tokens_used, llm_temperature, llm_max_tokens,
                retrieved_doc_ids_str, retrieval_top_k,
                use_documents, use_llm, processing_time_ms, error_message
            ))
            message_id = cursor.lastrowid
            
            # Update conversation timestamp
            conn.execute("""
                UPDATE conversations SET updated_at = ? WHERE id = ?
            """, (timestamp, conversation_id))
            
            conn.commit()
        
        logger.debug(f"Added RAG message {message_id} to conversation {conversation_id}")
        return message_id
    
    async def get_messages(self, conversation_id: str, user_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get messages from a conversation."""
        # First verify ownership
        conv = await self.get_conversation(conversation_id, user_id)
        if not conv:
            return []
        
        with self._connect() as conn:
            # To get recent messages with limit, we need to sort DESC first
            if limit:
                query = """
                    SELECT * FROM (
                        SELECT * FROM messages
                        WHERE conversation_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    ) ORDER BY created_at ASC
                """
                params = (conversation_id, limit)
            else:
                query = """
                    SELECT * FROM messages
                    WHERE conversation_id = ?
                    ORDER BY created_at ASC
                """
                params = (conversation_id,)
            
            rows = conn.execute(query, params).fetchall()
            
            messages = []
            for row in rows:
                msg = dict(row)
                
                # Parse JSON fields
                for key in ["retrieved_context", "embeddings_used", "sentiment_meta"]:
                    if msg.get(key) and isinstance(msg[key], str):
                        try:
                            msg[key] = json.loads(msg[key])
                        except json.JSONDecodeError:
                            logger.warning(f"Could not decode JSON for {key} in message {msg['id']}")
                            msg[key] = None
            
                if msg.get("retrieved_doc_ids") and isinstance(msg.get("retrieved_doc_ids"), str):
                    msg["retrieved_doc_ids"] = msg["retrieved_doc_ids"].split(",")
            
                messages.append(msg)
            
            return messages
    
    async def generate_title(self, conversation_id: str) -> str:
        """Auto-generate a conversation title from first messages."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT content FROM messages
                WHERE conversation_id = ? AND speaker = 'user'
                ORDER BY created_at ASC
                LIMIT 1
            """, (conversation_id,)).fetchone()
            
            if row:
                # Take first 50 characters of first user message
                content = row["content"]
                title = content[:50] + "..." if len(content) > 50 else content
                return title
            
            return "New Conversation"

    async def add_agent_message(
        self,
        conversation_id: str,
        speaker: str,
        content: str,
        user_query: Optional[str] = None,
        tools_used: Optional[List[str]] = None,
        steps: Optional[List[Dict[str, Any]]] = None,
        orchestrator_type: Optional[str] = None,
        processing_time_ms: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> int:
        """Add a message with agent pipeline logging to agent_messages table."""
        timestamp = datetime.utcnow().isoformat() + "Z"

        tools_used_json = json.dumps(tools_used) if tools_used else None
        steps_json = json.dumps(steps) if steps else None

        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO agent_messages (
                    conversation_id, speaker, content, created_at,
                    user_query, tools_used, steps, orchestrator_type,
                    processing_time_ms, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation_id, speaker, content, timestamp,
                user_query, tools_used_json, steps_json, orchestrator_type,
                processing_time_ms, error_message
            ))
            message_id = cursor.lastrowid

            conn.execute("""
                UPDATE conversations SET updated_at = ? WHERE id = ?
            """, (timestamp, conversation_id))

            conn.commit()

        logger.debug(f"Added agent message {message_id} to conversation {conversation_id}")
        return message_id

    async def get_agent_messages(
        self, conversation_id: str, user_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get agent messages from agent_messages table for a conversation."""
        conv = await self.get_conversation(conversation_id, user_id)
        if not conv:
            return []

        with self._connect() as conn:
            if limit:
                query = """
                    SELECT * FROM (
                        SELECT * FROM agent_messages
                        WHERE conversation_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    ) ORDER BY created_at ASC
                """
                params = (conversation_id, limit)
            else:
                query = """
                    SELECT * FROM agent_messages
                    WHERE conversation_id = ?
                    ORDER BY created_at ASC
                """
                params = (conversation_id,)

            rows = conn.execute(query, params).fetchall()

            messages = []
            for row in rows:
                msg = dict(row)
                for key in ["tools_used", "steps"]:
                    if msg.get(key) and isinstance(msg[key], str):
                        try:
                            msg[key] = json.loads(msg[key])
                        except json.JSONDecodeError:
                            msg[key] = None
                messages.append(msg)

            return messages

    async def add_crew_message(
        self,
        conversation_id: str,
        speaker: str,
        content: str,
        user_topic: Optional[str] = None,
        workflow_type: Optional[str] = None,
        agents_used: Optional[List[str]] = None,
        iterations: Optional[int] = None,
        processing_time_ms: Optional[int] = None,
        error_message: Optional[str] = None
    ) -> int:
        """Add a message with CrewAI workflow logging to crew_messages table."""
        timestamp = datetime.utcnow().isoformat() + "Z"
        agents_used_json = json.dumps(agents_used) if agents_used else None

        with self._connect() as conn:
            cursor = conn.execute("""
                INSERT INTO crew_messages (
                    conversation_id, speaker, content, created_at,
                    user_topic, workflow_type, agents_used, iterations,
                    processing_time_ms, error_message
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                conversation_id, speaker, content, timestamp,
                user_topic, workflow_type, agents_used_json, iterations,
                processing_time_ms, error_message
            ))
            message_id = cursor.lastrowid

            conn.execute("""
                UPDATE conversations SET updated_at = ? WHERE id = ?
            """, (timestamp, conversation_id))

            conn.commit()

        logger.debug(f"Added crew message {message_id} to conversation {conversation_id}")
        return message_id

    async def get_crew_messages(
        self, conversation_id: str, user_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get CrewAI messages from crew_messages table for a conversation."""
        conv = await self.get_conversation(conversation_id, user_id)
        if not conv:
            return []

        with self._connect() as conn:
            if limit:
                query = """
                    SELECT * FROM (
                        SELECT * FROM crew_messages
                        WHERE conversation_id = ?
                        ORDER BY created_at DESC
                        LIMIT ?
                    ) ORDER BY created_at ASC
                """
                params = (conversation_id, limit)
            else:
                query = """
                    SELECT * FROM crew_messages
                    WHERE conversation_id = ?
                    ORDER BY created_at ASC
                """
                params = (conversation_id,)

            rows = conn.execute(query, params).fetchall()

            messages = []
            for row in rows:
                msg = dict(row)
                if msg.get("agents_used") and isinstance(msg["agents_used"], str):
                    try:
                        msg["agents_used"] = json.loads(msg["agents_used"])
                    except json.JSONDecodeError:
                        msg["agents_used"] = None
                messages.append(msg)

            return messages
