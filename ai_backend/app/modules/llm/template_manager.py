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
                    messages TEXT DEFAULT '[]',
                    prompt_variables TEXT DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)
            
            # Check if messages column exists, if not add it
            cursor = conn.execute("PRAGMA table_info(templates)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'messages' not in columns:
                conn.execute("ALTER TABLE templates ADD COLUMN messages TEXT DEFAULT '[]'")
                conn.commit()
                logger.info("Added messages column to templates table")
            
            if 'prompt_variables' not in columns:
                conn.execute("ALTER TABLE templates ADD COLUMN prompt_variables TEXT DEFAULT ''")
                conn.commit()
                logger.info("Added prompt_variables column to templates table")
            
            conn.commit()

    def _seed_default_templates(self):
        """Seed default templates with message arrays."""
        templates_to_seed = [
            {
                "name": "pirate_template",
                "messages": [
                    {"role": "system", "content": "You are a pirate. Always respond like a pirate with 'Ahoy!' and pirate language. Use words like 'matey', 'arrr', 'ye', 'me hearty' in every response. Never break character - you are always a pirate."},
                    {"role": "user", "content": "{user_question}"}
                ],
                "variables": "user_question"
            },
            {
                "name": "json_bot_template", 
                "messages": [
                    {"role": "system", "content": "You only respond with valid JSON. No extra text. Always format your response as proper JSON."},
                    {"role": "user", "content": "{user_question}"}
                ],
                "variables": "user_question"
            },
            {
                "name": "enterprise_assistant",
                "messages": [
                    {"role": "system", "content": "You are a professional enterprise assistant. Use the provided context to answer questions accurately. User role: {user_role}, Department: {department}"},
                    {"role": "user", "content": "Context: {source_docs}\n\nQuestion: {user_question}"}
                ],
                "variables": "user_role|department|source_docs|user_question"
            },
            {
                "name": "personalized_chat",
                "messages": [
                    {"role": "system", "content": "You are a friendly but professional enterprise assistant. Personalize responses using the user profile when helpful. User profile: {user_profile_summary}, Role: {user_role}, Department: {department}"},
                    {"role": "user", "content": "Context: {source_docs}\n\nConversation history: {history}\n\nQuestion: {user_question}\n\nStyle: Address the user naturally when appropriate, stay professional, do not exceed {max_tokens} tokens"}
                ],
                "variables": "user_profile_summary|user_role|department|source_docs|history|user_question|max_tokens"
            }
        ]
        
        for template_data in templates_to_seed:
            # Check if exists
            try:
                if self.get_template(template_data["name"]):
                    continue
            except Exception:
                pass
            
            try:
                logger.info(f"Seeding template '{template_data['name']}' with message array")
                self.create_template(
                    name=template_data["name"],
                    content="",  # Legacy content field
                    messages=template_data["messages"],
                    prompt_variables=template_data["variables"]
                )
            except Exception as e:
                logger.error(f"Failed to seed template '{template_data['name']}': {e}")

    def _get_timestamp(self) -> str:
        return datetime.utcnow().isoformat() + "Z"

    def create_template(self, name: str, content: str = '', messages: list = None, prompt_variables: str = '') -> Dict[str, Any]:
        """Create a new template with messages array."""
        import json
        timestamp = self._get_timestamp()
        
        # Convert messages to JSON string
        messages_json = json.dumps(messages or [])
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    INSERT INTO templates (name, content, messages, prompt_variables, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (name, content, messages_json, prompt_variables, timestamp, timestamp))
                conn.commit()
                
                return self.get_template(name)
        except sqlite3.IntegrityError:
            raise ValueError(f"Template with name '{name}' already exists")

    def get_template(self, name: str) -> Optional[Dict[str, Any]]:
        """Get a template by name."""
        import json
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM templates WHERE name = ?", (name,)).fetchone()
            if row:
                template = dict(row)
                # Parse messages JSON
                try:
                    template['messages'] = json.loads(template.get('messages', '[]'))
                except (json.JSONDecodeError, TypeError):
                    template['messages'] = []
                return template
            return None

    def list_templates(self) -> List[Dict[str, Any]]:
        """List all templates."""
        import json
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute("SELECT * FROM templates ORDER BY name ASC").fetchall()
            templates = []
            for row in rows:
                template = dict(row)
                # Parse messages JSON
                try:
                    template['messages'] = json.loads(template.get('messages', '[]'))
                except (json.JSONDecodeError, TypeError):
                    template['messages'] = []
                templates.append(template)
            return templates

    def update_template(self, name: str, content: str = None, messages: list = None, prompt_variables: str = None) -> Optional[Dict[str, Any]]:
        """Update an existing template."""
        import json
        timestamp = self._get_timestamp()
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Build update query dynamically
            updates = []
            params = []
            
            if content is not None:
                updates.append("content = ?")
                params.append(content)
            
            if messages is not None:
                updates.append("messages = ?")
                params.append(json.dumps(messages))
            
            if prompt_variables is not None:
                updates.append("prompt_variables = ?")
                params.append(prompt_variables)
            
            updates.append("updated_at = ?")
            params.append(timestamp)
            params.append(name)
            
            query = f"UPDATE templates SET {', '.join(updates)} WHERE name = ?"
            cursor = conn.execute(query, params)
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
