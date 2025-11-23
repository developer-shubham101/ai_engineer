from __future__ import annotations

import logging
import sqlite3
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import centralized paths and utilities
from app.services.utility import (
    BASE_DIR,
    DATA_DIR,
    get_config_path,
    build_tone_guidance,
)

# new import for sentiment classifier
from app.services.sentiment_classifier import get_global_sentiment

logger = logging.getLogger(__name__)

# DB PATHS
DB_PATH = DATA_DIR / "support_sessions.db"

MAX_HISTORY_TURNS = 5
_DB_INITIALIZED = False


# ---------------------------------------------------------
# DB INIT
# ---------------------------------------------------------
def init_support_chat_db(reset_on_start: bool = False) -> None:
    """
    Initializes the SQLite DB.
    Creates:
        sessions
        messages
        session_profiles
    Note: messages table now stores sentiment metadata columns.
    """
    global _DB_INITIALIZED
    if _DB_INITIALIZED:
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if reset_on_start and DB_PATH.exists():
        try:
            DB_PATH.unlink()
            logger.info("Support chat DB reset for fresh session state.")
        except OSError as exc:
            logger.warning("Unable to reset support chat DB: %s", exc)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                role TEXT,
                department TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """)

        # messages now include sentiment and tone metadata
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

        # NEW TABLE — stores onboarding/profile info
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

    _DB_INITIALIZED = True


# ---------------------------------------------------------
# DB CONNECTOR
# ---------------------------------------------------------
def _connect() -> sqlite3.Connection:
    init_support_chat_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------
# SESSION MANAGEMENT
# ---------------------------------------------------------
def create_session(session_id: Optional[str], role: Optional[str], department: Optional[str]) -> str:
    sid = session_id or f"sess_{uuid.uuid4().hex}"
    timestamp = datetime.utcnow().isoformat() + "Z"

    with _connect() as conn:
        existing = conn.execute("SELECT id FROM sessions WHERE id=?", (sid,)).fetchone()
        if existing:
            raise ValueError(f"Session '{sid}' already exists.")

        conn.execute("""
            INSERT INTO sessions (id, role, department, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?)
        """, (sid, role, department, timestamp, timestamp))
        conn.commit()

    logger.info("Created support session %s (role=%s dept=%s)", sid, role, department)
    return sid


def touch_session(session_id: str, role: Optional[str], department: Optional[str]) -> None:
    timestamp = datetime.utcnow().isoformat() + "Z"

    with _connect() as conn:
        updated = conn.execute("""
            UPDATE sessions
            SET role = COALESCE(?, role),
                department = COALESCE(?, department),
                updated_at = ?
            WHERE id = ?
        """, (role, department, timestamp, session_id))

        if updated.rowcount == 0:
            raise ValueError(f"Session '{session_id}' not found.")

        conn.commit()


def end_session(session_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        deleted = conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        if deleted.rowcount == 0:
            raise ValueError(f"Session '{session_id}' not found.")
        conn.commit()

    logger.info("Ended support session %s", session_id)


def session_exists(session_id: str) -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT id FROM sessions WHERE id=?", (session_id,)).fetchone()
        return bool(row)


# ---------------------------------------------------------
# MESSAGE STORAGE
# ---------------------------------------------------------
def store_message(session_id: str, speaker: str, content: str) -> int:
    """
    Store a message and (for user messages) compute & store sentiment/tone metadata.
    Returns the inserted message id.
    """
    timestamp = datetime.utcnow().isoformat() + "Z"

    with _connect() as conn:
        cur = conn.cursor()
        cur.execute("""
            INSERT INTO messages (session_id, speaker, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (session_id, speaker, content, timestamp))
        message_id = cur.lastrowid

        # If speaker is user, compute sentiment/tone and update row
        # Always store sentiment/tone/metadata - use defaults if classifier fails
        if speaker.lower() == "user":
            sentiment = "unknown"
            tone = "neutral"
            meta_json = json.dumps({"sentiment": {"unknown": 1.0}, "tone": {"neutral": 1.0}})
            
            try:
                classifier = get_global_sentiment()
                res = classifier.predict_single(content)
                sentiment = res.get("sentiment", "unknown")
                tone = res.get("tone", "neutral")
                meta_json = json.dumps(res.get("proba", {"sentiment": {"unknown": 1.0}, "tone": {"neutral": 1.0}}))
            except Exception as e:
                # Never fail storing a message due to classifier errors; use defaults
                logger.warning("Sentiment classification failed for message_id=%s: %s. Using defaults.", message_id, e)
            
            # Always update with sentiment/tone/metadata (even if defaults)
            try:
                cur.execute("""
                    UPDATE messages
                    SET sentiment=?, tone=?, sentiment_meta=?
                    WHERE id=?
                """, (sentiment, tone, meta_json, message_id))
            except Exception as e:
                logger.warning("Failed to update sentiment metadata for message_id=%s: %s", message_id, e)

        conn.commit()
        return message_id


def fetch_recent_messages(session_id: str, limit: int = MAX_HISTORY_TURNS) -> List[Dict[str, str]]:
    """
    Fetch recent messages for a session.
    Always returns sentiment and tone fields (defaults to None if not set).
    """
    with _connect() as conn:
        rows = conn.execute("""
            SELECT speaker, content, created_at, sentiment, tone, sentiment_meta
            FROM messages
            WHERE session_id=?
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, limit)).fetchall()

    # reverse to chronological order
    messages = [dict(row) for row in reversed(rows)]
    
    # Ensure all messages have sentiment, tone, and sentiment_meta fields
    # If sentiment_meta exists as JSON string, convert to dict
    for m in messages:
        # Ensure fields exist (default to None if missing)
        if "sentiment" not in m:
            m["sentiment"] = None
        if "tone" not in m:
            m["tone"] = None
        if "sentiment_meta" not in m:
            m["sentiment_meta"] = None
        
        # Parse sentiment_meta JSON if present
        if m.get("sentiment_meta"):
            try:
                if isinstance(m["sentiment_meta"], str):
                    m["sentiment_meta"] = json.loads(m["sentiment_meta"])
            except Exception:
                # If parsing fails, set to empty dict
                m["sentiment_meta"] = {}
        else:
            m["sentiment_meta"] = {}
    
    return messages


# ---------------------------------------------------------
# RENDER HISTORY
# ---------------------------------------------------------
def render_history(messages: List[Dict[str, str]]) -> str:
    if not messages:
        return "No previous conversation."

    lines = []
    for msg in messages:
        stamp = msg.get("created_at", "")
        speaker = msg.get("speaker", "").upper()
        content = msg.get("content", "")
        # include tone indicator for user messages if present
        tone = msg.get("tone")
        tone_suffix = f" [{tone}]" if tone else ""
        lines.append(f"[{stamp}] {speaker}{tone_suffix}: {content}")

    return "\n".join(lines)


# ---------------------------------------------------------
# PROFILE MANAGEMENT (NEW)
# ---------------------------------------------------------
def set_profile_value(session_id: str, key: str, value: str) -> None:
    with _connect() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO session_profiles (session_id, key, value)
            VALUES (?, ?, ?)
        """, (session_id, key, value))
        conn.commit()


def get_profile_value(session_id: str, key: str) -> Optional[str]:
    with _connect() as conn:
        row = conn.execute("""
            SELECT value FROM session_profiles
            WHERE session_id=? AND key=?
        """, (session_id, key)).fetchone()

    return row["value"] if row else None


def get_full_profile(session_id: str) -> Dict[str, str]:
    with _connect() as conn:
        rows = conn.execute("""
            SELECT key, value FROM session_profiles
            WHERE session_id=?
        """, (session_id,)).fetchall()

    return {row["key"]: row["value"] for row in rows}


def load_onboarding_fields() -> List[Dict[str, str]]:
    config_path = get_config_path("onboarding_fields.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_next_missing_profile_key(session_id: str) -> Optional[Dict[str, str]]:
    fields = load_onboarding_fields()
    profile = get_full_profile(session_id)

    for field in fields:
        if field["key"] not in profile:
            return field

    return None


# ---------------------------------------------------------
# TONE GUIDANCE HELPER
# ---------------------------------------------------------
# build_tone_guidance is now imported from utility.py


# ---------------------------------------------------------
# BUILD PROMPT PREFIX
# ---------------------------------------------------------
def build_prompt_prefix(requester: Dict[str, Optional[str]],
                        history: List[Dict[str, str]],
                        category: Optional[str]) -> str:
    role = requester.get("role") or "Guest"
    dept = requester.get("department") or "General"
    cat = (category or "General").title()
    history_text = render_history(history)

    # extract last user tone from history (if present)
    last_user_tone = None
    # iterate history backwards to find last user message with tone
    for msg in reversed(history):
        if msg.get("speaker", "").lower() == "user" and msg.get("tone"):
            last_user_tone = msg.get("tone")
            break

    tone_guidance = build_tone_guidance(last_user_tone)

    return (
        f"You are a {cat} support assistant.\n"
        f"User Role: {role}\n"
        f"User Department: {dept}\n\n"
        f"Conversation Tone Guidance:\n{tone_guidance}\n\n"
        f"Conversation History:\n{history_text}\n"
    )


# ---------------------------------------------------------
# SENTIMENT / TONE ANALYTICS
# ---------------------------------------------------------
def get_sentiment_stats() -> Dict[str, Dict]:
    """
    Return simple stats:
      - tone_counts: {tone: count}
      - sentiment_counts: {sentiment: count}
      - tone_by_department: list of {department, tone, count}
    """
    with _connect() as conn:
        q1 = conn.execute("""
            SELECT tone, COUNT(*) as cnt FROM messages
            WHERE speaker='user' AND tone IS NOT NULL
            GROUP BY tone
        """).fetchall()
        q2 = conn.execute("""
            SELECT sentiment, COUNT(*) as cnt FROM messages
            WHERE speaker='user' AND sentiment IS NOT NULL
            GROUP BY sentiment
        """).fetchall()
        q3 = conn.execute("""
            SELECT s.department as department, m.tone as tone, COUNT(*) as cnt
            FROM messages m
            JOIN sessions s ON s.id = m.session_id
            WHERE m.speaker='user' AND m.tone IS NOT NULL
            GROUP BY s.department, m.tone
        """).fetchall()

    tone_counts = {row["tone"]: row["cnt"] for row in q1}
    sentiment_counts = {row["sentiment"]: row["cnt"] for row in q2}
    tone_by_department = [{"department": row["department"], "tone": row["tone"], "count": row["cnt"]} for row in q3]

    return {
        "tone_counts": tone_counts,
        "sentiment_counts": sentiment_counts,
        "tone_by_department": tone_by_department
    }
