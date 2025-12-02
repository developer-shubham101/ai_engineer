"""Session management implementation."""

from __future__ import annotations

import json
import logging
import sqlite3
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from app.modules.llm.prompt_builder import build_tone_guidance
# NOTE: get_global_sentiment from legacy services is removed. Using a placeholder for sentiment analysis.
from .interfaces import ISessionManager
from ..config.settings import settings


logger = logging.getLogger(__name__)


# Global constants moved to class attributes defaults
_DEFAULT_DB_PATH = settings.DATABASE_DIR / "support_sessions.db"
_DEFAULT_MAX_HISTORY_TURNS = 5


class SQLiteSessionManager(ISessionManager):
    """
    SQLite-based session management implementation, incorporating all
    support chat features from the original support_chat.py module.
    """

    def __init__(self, db_path: Optional[Any] = None, max_history_turns: int = _DEFAULT_MAX_HISTORY_TURNS):
        # Use the provided path or the default
        self.db_path = db_path or _DEFAULT_DB_PATH
        self.max_history_turns = max_history_turns
        self._db_initialized = False

        # Ensure database is initialized on service startup
        self.init_support_chat_db()


    # ---------------------------------------------------------
    # DB INIT (Function name kept: init_support_chat_db)
    # ---------------------------------------------------------
    def init_support_chat_db(self, reset_on_start: bool = False) -> None:
        """
        Initializes the SQLite DB with the required tables: sessions, messages, session_profiles.
        (Matches original function name and logic)
        """
        if self._db_initialized:
            return

        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if reset_on_start and self.db_path.exists():
            try:
                self.db_path.unlink()
                logger.info("Support chat DB reset for fresh session state.")
            except OSError as exc:
                logger.warning("Unable to reset support chat DB: %s", exc)

        with sqlite3.connect(self.db_path) as conn:
            # 1. sessions table (Structure preserved)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    role TEXT,
                    department TEXT,
                    created_at TEXT,
                    updated_at TEXT
                )
            """)

            # 2. messages table (Structure preserved, including sentiment/tone columns)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    speaker TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT,
                    sentiment TEXT,
                    tone TEXT,
                    sentiment_meta TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)

            # 3. session_profiles table (Structure preserved for onboarding data)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_profiles (
                    session_id TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    PRIMARY KEY (session_id, key),
                    FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
                )
            """)

            conn.commit()

        self._db_initialized = True


    # ---------------------------------------------------------
    # DB CONNECTOR (Function name kept: _connect)
    # ---------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        """Internal connection helper."""
        self.init_support_chat_db() # Ensure DB is initialized
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn


    # ---------------------------------------------------------
    # SESSION MANAGEMENT (Functions kept same name/signature)
    # ---------------------------------------------------------
    def create_session(self, session_id: Optional[str], role: Optional[str], department: Optional[str]) -> str:
        """Create a new support chat session."""
        sid = session_id or f"sess_{uuid.uuid4().hex}"
        timestamp = datetime.utcnow().isoformat() + "Z"

        with self._connect() as conn:
            existing = conn.execute("SELECT id FROM sessions WHERE id=?", (sid,)).fetchone()
            if existing:
                raise ValueError(f"Session '{sid}' already exists.")

            conn.execute("""
                INSERT INTO sessions (id, role, department, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
            """, (sid, role, department, timestamp, timestamp))
            conn.commit()

        logger.info(f"SUPPORT_SESSION_CREATED: {sid} for role={role}, department={department}")
        return sid


    async def touch_session(self, session_id: str) -> None:
        """Update session metadata/timestamp."""
        timestamp = datetime.utcnow().isoformat() + "Z"

        with self._connect() as conn:
            updated = conn.execute("""
                UPDATE sessions
                SET updated_at = ?
                WHERE id = ?
            """, (timestamp, session_id))

            if updated.rowcount == 0:
                raise ValueError(f"Session '{session_id}' not found.")

            conn.commit()


    def end_session(self, session_id: str) -> None:
        """End and delete a support chat session and its data."""
        with self._connect() as conn:
            # Delete messages/profiles (ON DELETE CASCADE should handle, but explicit is safer)
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM session_profiles WHERE session_id=?", (session_id,))
            deleted = conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
            if deleted.rowcount == 0:
                raise ValueError(f"Session '{session_id}' not found.")
            conn.commit()

        logger.info(f"SUPPORT_SESSION_ENDED: {session_id}")


    def session_exists(self, session_id: str) -> bool:
        """Check if a session ID exists."""
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone()
            return bool(row)


    # ---------------------------------------------------------
    # MESSAGE STORAGE (Functions kept same name/signature)
    # ---------------------------------------------------------
    def store_message(self, session_id: str, speaker: str, content: str) -> int:
        """Store a message and compute sentiment/tone."""
        timestamp = datetime.utcnow().isoformat() + "Z"

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO messages (session_id, speaker, content, created_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, speaker, content, timestamp))
            message_id = cur.lastrowid

            # If speaker is user, compute sentiment/tone and update row
            if speaker.lower() == "user":
                sentiment = "neutral" # Placeholder
                tone = "neutral" # Placeholder
                meta_json = json.dumps({"sentiment": {"neutral": 1.0}, "tone": {"neutral": 1.0}})

                # Original sentiment classification logic removed, using placeholders.
                # try:
                #     classifier = get_global_sentiment()
                #     res = classifier.predict_single(content)
                #     sentiment = res.get("sentiment", "unknown")
                #     tone = res.get("tone", "neutral")
                #     meta_json = json.dumps(res.get("proba", {}))
                # except Exception as e:
                #     logger.warning(f"Sentiment classification failed for message {message_id}: {e}")

                # Always update with sentiment/tone/metadata
                try:
                    cur.execute("""
                        UPDATE messages
                        SET sentiment=?, tone=?, sentiment_meta=?
                        WHERE id=?
                    """, (sentiment, tone, meta_json, message_id))
                except Exception as e:
                    logger.error(f"Failed to update message {message_id} with sentiment metadata: {e}")

            conn.commit()
            return message_id


    def fetch_recent_messages(self, session_id: str, limit: int = _DEFAULT_MAX_HISTORY_TURNS) -> List[Dict[str, Any]]:
        """Fetch recent messages for a session."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT speaker, content, created_at, sentiment, tone, sentiment_meta
                FROM messages
                WHERE session_id=?
                ORDER BY id DESC
                LIMIT ?
            """, (session_id, limit)).fetchall()

        # Reverse to chronological order and ensure fields are present
        messages = [dict(row) for row in reversed(rows)]

        for m in messages:
            # Parse sentiment_meta JSON if present
            meta_str = m.get("sentiment_meta")
            m["sentiment_meta"] = {}
            if meta_str and isinstance(meta_str, str):
                try:
                    m["sentiment_meta"] = json.loads(meta_str)
                except Exception:
                    m["sentiment_meta"] = {}

        return messages


    # ---------------------------------------------------------
    # RENDER HISTORY (Function name kept: render_history)
    # ---------------------------------------------------------
    def render_history(self, messages: List[Dict[str, Any]]) -> str:
        """Render message history into a single string."""
        if not messages:
            return "No previous conversation."

        lines = []
        for msg in messages:
            stamp = msg.get("created_at", "")
            speaker = msg.get("speaker", "").upper()
            content = msg.get("content", "")
            tone = msg.get("tone")
            # Only include tone indicator for user messages if present
            tone_suffix = f" [{tone}]" if speaker == "USER" and tone else ""
            lines.append(f"[{stamp}] {speaker}{tone_suffix}: {content}")

        return "\n".join(lines)


    # ---------------------------------------------------------
    # PROFILE MANAGEMENT (Functions kept same name/signature)
    # ---------------------------------------------------------
    def set_profile_value(self, session_id: str, key: str, value: str) -> None:
        """Set a single profile key/value for the session."""
        with self._connect() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO session_profiles (session_id, key, value)
                VALUES (?, ?, ?)
            """, (session_id, key, value))
            conn.commit()


    def get_profile_value(self, session_id: str, key: str) -> Optional[str]:
        """Get a single profile value for the session."""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT value FROM session_profiles
                WHERE session_id=? AND key=?
            """, (session_id, key)).fetchone()

        return row["value"] if row else None


    def get_full_profile(self, session_id: str) -> Dict[str, str]:
        """Get the full profile dictionary for the session."""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT key, value FROM session_profiles
                WHERE session_id=?
            """, (session_id,)).fetchall()

        return {row["key"]: row["value"] for row in rows}


    def load_onboarding_fields(self) -> List[Dict[str, str]]:
        """Load configured onboarding fields (from JSON file)."""
        config_path = settings.CONFIG_DIR / "onboarding_fields.json"
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)


    def get_next_missing_profile_key(self, session_id: str) -> Optional[Dict[str, str]]:
        """Get the next missing profile key for onboarding."""
        fields = self.load_onboarding_fields()
        profile = self.get_full_profile(session_id)

        for field in fields:
            if field["key"] not in profile:
                return field

        return None

    # ---------------------------------------------------------
    # SENTIMENT / TONE ANALYTICS (Function name kept: get_sentiment_stats)
    # ---------------------------------------------------------
    def get_sentiment_stats(self) -> Dict[str, Dict]:
        """
        Return simple stats: {tone_counts, sentiment_counts, tone_by_department}
        """
        with self._connect() as conn:
            q1 = conn.execute("""
                SELECT tone, COUNT(*) as cnt FROM messages
                WHERE speaker='user' AND tone IS NOT NULL
                GROUP BY tone
            """ ).fetchall()
            q2 = conn.execute("""
                SELECT sentiment, COUNT(*) as cnt FROM messages
                WHERE speaker='user' AND sentiment IS NOT NULL
                GROUP BY sentiment
            """ ).fetchall()
            q3 = conn.execute("""
                SELECT s.department as department, m.tone as tone, COUNT(*) as cnt
                FROM messages m
                JOIN sessions s ON s.id = m.session_id
                WHERE m.speaker='user' AND m.tone IS NOT NULL
                GROUP BY s.department, m.tone
            """ ).fetchall()

        tone_counts = {row["tone"]: row["cnt"] for row in q1}
        sentiment_counts = {row["sentiment"]: row["cnt"] for row in q2}
        tone_by_department = [{"department": row["department"], "tone": row["tone"], "count": row["cnt"]} for row in q3]

        return {
            "tone_counts": tone_counts,
            "sentiment_counts": sentiment_counts,
            "tone_by_department": tone_by_department
        }
