"""Session management implementation."""

import sqlite3
import json
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List
import logging

from .interfaces import ISessionManager
from ..config.settings import settings

logger = logging.getLogger(__name__)


class SQLiteSessionManager(ISessionManager):
    """SQLite-based session management implementation."""
    
    def __init__(self):
        self.db_path = settings.DATABASE_DIR / "support_sessions.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize session database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT,
                    role TEXT,
                    department TEXT,
                    metadata TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    speaker TEXT,
                    content TEXT,
                    metadata TEXT,
                    sentiment TEXT,
                    tone TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id 
                ON messages (session_id)
            """)
    
    async def create_session(self, user_id: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Create new session."""
        session_id = str(uuid.uuid4())
        metadata_json = json.dumps(metadata or {})
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO sessions (session_id, user_id, metadata)
                VALUES (?, ?, ?)
            """, (session_id, user_id, metadata_json))
        
        logger.info(f"Session created: {session_id} for user: {user_id}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            )
            row = cursor.fetchone()
            
            if row:
                session = dict(row)
                if session["metadata"]:
                    session["metadata"] = json.loads(session["metadata"])
                return session
            return None
    
    async def update_session(self, session_id: str, data: Dict[str, Any]) -> bool:
        """Update session data."""
        with sqlite3.connect(self.db_path) as conn:
            # Update specific fields
            if "role" in data:
                conn.execute(
                    "UPDATE sessions SET role = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                    (data["role"], session_id)
                )
            
            if "department" in data:
                conn.execute(
                    "UPDATE sessions SET department = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                    (data["department"], session_id)
                )
            
            if "metadata" in data:
                metadata_json = json.dumps(data["metadata"])
                conn.execute(
                    "UPDATE sessions SET metadata = ?, updated_at = CURRENT_TIMESTAMP WHERE session_id = ?",
                    (metadata_json, session_id)
                )
        
        return True
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session and all its messages."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        
        logger.info(f"Session deleted: {session_id}")
        return True
    
    async def store_message(self, session_id: str, speaker: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store message in session history."""
        metadata_json = json.dumps(metadata or {})
        
        # Simple sentiment analysis (placeholder)
        sentiment = self._analyze_sentiment(content)
        tone = self._analyze_tone(content)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO messages (session_id, speaker, content, metadata, sentiment, tone)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, speaker, content, metadata_json, sentiment, tone))
        
        return True
    
    async def get_messages(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get session message history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM messages 
                WHERE session_id = ? 
                ORDER BY created_at DESC 
                LIMIT ?
            """, (session_id, limit))
            
            messages = []
            for row in cursor.fetchall():
                message = dict(row)
                if message["metadata"]:
                    message["metadata"] = json.loads(message["metadata"])
                messages.append(message)
            
            return list(reversed(messages))  # Return in chronological order
    
    async def clear_messages(self, session_id: str) -> bool:
        """Clear session message history."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        
        logger.info(f"Messages cleared for session: {session_id}")
        return True
    
    def _analyze_sentiment(self, content: str) -> str:
        """Simple sentiment analysis (placeholder)."""
        # In production, use proper sentiment analysis
        positive_words = ["good", "great", "excellent", "happy", "pleased"]
        negative_words = ["bad", "terrible", "awful", "sad", "angry"]
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def _analyze_tone(self, content: str) -> str:
        """Simple tone analysis (placeholder)."""
        # In production, use proper tone analysis
        if "?" in content:
            return "questioning"
        elif "!" in content:
            return "emphatic"
        elif any(word in content.lower() for word in ["please", "thank", "sorry"]):
            return "polite"
        else:
            return "neutral"