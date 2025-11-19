from __future__ import annotations

import logging
import sqlite3
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from app.services.rag_local_service import BASE_DIR

logger = logging.getLogger(__name__)

# DB PATHS
DATA_DIR = BASE_DIR / "data"
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

        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                speaker TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT,
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
def store_message(session_id: str, speaker: str, content: str) -> None:
    timestamp = datetime.utcnow().isoformat() + "Z"

    with _connect() as conn:
        conn.execute("""
            INSERT INTO messages (session_id, speaker, content, created_at)
            VALUES (?, ?, ?, ?)
        """, (session_id, speaker, content, timestamp))
        conn.commit()


def fetch_recent_messages(session_id: str, limit: int = MAX_HISTORY_TURNS) -> List[Dict[str, str]]:
    with _connect() as conn:
        rows = conn.execute("""
            SELECT speaker, content, created_at
            FROM messages
            WHERE session_id=?
            ORDER BY id DESC
            LIMIT ?
        """, (session_id, limit)).fetchall()

    return [dict(row) for row in reversed(rows)]


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
        lines.append(f"[{stamp}] {speaker}: {content}")

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
    config_path = BASE_DIR / "config" / "onboarding_fields.json"
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
# BUILD PROMPT PREFIX
# ---------------------------------------------------------
def build_prompt_prefix(requester: Dict[str, Optional[str]],
                        history: List[Dict[str, str]],
                        category: Optional[str]) -> str:
    role = requester.get("role") or "Guest"
    dept = requester.get("department") or "General"
    cat = (category or "General").title()
    history_text = render_history(history)

    return (
        f"You are a {cat} support assistant.\n"
        f"User Role: {role}\n"
        f"User Department: {dept}\n\n"
        f"Conversation History:\n{history_text}\n"
    )
