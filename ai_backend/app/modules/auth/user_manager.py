"""User management implementation."""

import sqlite3
import json
from typing import Optional, Dict, Any
import logging
from pathlib import Path

from .interfaces import IUserManager
from ..config.settings import settings

logger = logging.getLogger(__name__)


class SQLiteUserManager(IUserManager):
    """SQLite-based user management implementation."""
    
    def __init__(self):
        self.db_path = settings.DATABASE_DIR / "users.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize user database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    role TEXT NOT NULL,
                    department TEXT,
                    profile TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_metadata (
                    user_id TEXT,
                    key TEXT,
                    value TEXT,
                    PRIMARY KEY (user_id, key),
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            """)
            
            # Create default admin user if not exists
            self._create_default_users(conn)
    
    def _create_default_users(self, conn):
        """Create default users for testing."""
        default_users = [
            {
                "user_id": "admin",
                "username": "admin",
                "password": "admin123",
                "role": "SuperAdmin",
                "department": "Admin",
                "profile": json.dumps({"name": "Administrator", "position": "System Admin"})
            },
            {
                "user_id": "employee1",
                "username": "employee1",
                "password": "password123",
                "role": "Employee",
                "department": "Engineering",
                "profile": json.dumps({"name": "John Doe", "position": "Software Engineer"})
            }
        ]
        
        for user in default_users:
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO users 
                    (user_id, username, password, role, department, profile)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (user["user_id"], user["username"], user["password"], 
                     user["role"], user["department"], user["profile"]))
            except sqlite3.Error as e:
                logger.error(f"Error creating default user {user['username']}: {e}")
    
    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM users WHERE user_id = ?", (user_id,)
            )
            row = cursor.fetchone()
            
            if row:
                user = dict(row)
                if user["profile"]:
                    user["profile"] = json.loads(user["profile"])
                return user
            return None
    
    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM users WHERE username = ?", (username,)
            )
            row = cursor.fetchone()
            
            if row:
                user = dict(row)
                if user["profile"]:
                    user["profile"] = json.loads(user["profile"])
                return user
            return None
    
    async def create_user(self, user_data: Dict[str, Any]) -> str:
        """Create new user."""
        user_id = user_data["user_id"]
        profile_json = json.dumps(user_data.get("profile", {}))
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO users (user_id, username, password, role, department, profile)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (user_id, user_data["username"], user_data["password"],
                 user_data["role"], user_data.get("department"), profile_json))
        
        logger.info(f"User created: {user_data['username']}")
        return user_id
    
    async def update_user(self, user_id: str, user_data: Dict[str, Any]) -> bool:
        """Update user data."""
        profile_json = json.dumps(user_data.get("profile", {})) if "profile" in user_data else None
        
        with sqlite3.connect(self.db_path) as conn:
            if profile_json:
                conn.execute("""
                    UPDATE users SET role = ?, department = ?, profile = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (user_data.get("role"), user_data.get("department"), profile_json, user_id))
            else:
                conn.execute("""
                    UPDATE users SET role = ?, department = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = ?
                """, (user_data.get("role"), user_data.get("department"), user_id))
        
        logger.info(f"User updated: {user_id}")
        return True
    
    async def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM user_metadata WHERE user_id = ?", (user_id,))
            conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
        
        logger.info(f"User deleted: {user_id}")
        return True
    
    async def get_user_metadata(self, user_id: str, key: str) -> Any:
        """Get user metadata value."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT value FROM user_metadata WHERE user_id = ? AND key = ?",
                (user_id, key)
            )
            row = cursor.fetchone()
            
            if row:
                try:
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    return row[0]
            return None
    
    async def set_user_metadata(self, user_id: str, key: str, value: Any) -> bool:
        """Set user metadata value."""
        value_json = json.dumps(value) if not isinstance(value, str) else value
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO user_metadata (user_id, key, value)
                VALUES (?, ?, ?)
            """, (user_id, key, value_json))
        
        return True