"""Template management service for prompts."""

import sqlite3
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

from ..config.settings import settings

logger = logging.getLogger(__name__)

class TemplateManager:
    """Manages prompt templates using SQLite."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (settings.DATABASE_DIR / "prompts.db")
        self._init_db()
        self._seed_default_templates()

    def _init_db(self):
        """Initialize the templates database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            conn.commit()

    def _seed_default_templates(self):
        """Seed default templates from file system if they don't exist."""
        default_template_name = "personalized_chat"
        
        # Check if exists
        try:
            if self.get_template(default_template_name):
                return
        except Exception:
            pass
            
        # Load from file
        try:
            template_path = Path("app/modules/llm/prompt_templates/personalized_chat.txt")
            if template_path.exists():
                with open(template_path, "r") as f:
                    content = f.read()
                
                logger.info(f"Seeding default template '{default_template_name}' from file")
                self.create_template(default_template_name, content)
            else:
                logger.warning(f"Default template file not found at {template_path}")
        except Exception as e:
            logger.error(f"Failed to seed default template: {e}")

    def _get_timestamp(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def create_template(self, name: str, content: str) -> Dict[str, Any]:
        """Create a new template."""
        timestamp = self._get_timestamp()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    INSERT INTO templates (name, content, created_at, updated_at)
                    VALUES (?, ?, ?, ?)
                """, (name, content, timestamp, timestamp))
                conn.commit()
                
                return self.get_template(name)
        except sqlite3.IntegrityError:
            raise ValueError(f"Template with name '{name}' already exists")

    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a template by name."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM templates WHERE name = ?", (name,)).fetchone()
            if row:
                return dict(row)
            return None

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all templates."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM templates ORDER BY name ASC").fetchall()
            return [dict(row) for row in rows]

    def update_template(self, name: str, content: str) -> Optional[Dict[str, Any]]:
        """Update an existing template."""
        timestamp = self._get_timestamp()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                UPDATE templates 
                SET content = ?, updated_at = ?
                WHERE name = ?
            """, (content, timestamp, name))
            conn.commit()
            
            if cursor.rowcount > 0:
                return self.get_template(name)
            return None

    def delete_template(self, name: str) -> bool:
        """Delete a template."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM templates WHERE name = ?", (name,))
            conn.commit()
            return cursor.rowcount > 0
