"""User management implementation."""

from __future__ import annotations

import json
import logging
import sqlite3
import bcrypt
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, List

from .interfaces import IUserManager
# NOTE: Assuming these imports are available in your environment structure

from ..config.settings import settings

logger = logging.getLogger(__name__)

# Dummy users for seeding (moved from user_service.py)
DUMMY_USERS = [
    {
        "username": "admin",
        "password": "admin123",
        "user_id": "u_admin_1",
        "role": "SuperAdmin",
        "department": "Executive"
    },
    {
        "username": "hr_manager",
        "password": "hr123",
        "user_id": "u_hr_1",
        "role": "HR",
        "department": "HR"
    },
    {
        "username": "manager",
        "password": "mgr123",
        "user_id": "u_mgr_1",
        "role": "Manager",
        "department": "Engineering"
    },
    {
        "username": "employee",
        "password": "emp123",
        "user_id": "u_emp_1",
        "role": "Employee",
        "department": "Sales"
    }
]

# Database path (moved from user_service.py)
USER_DB_PATH = settings.DATABASE_DIR / "users.db"


class SQLiteUserManager(IUserManager):
    """SQLite-based user management implementation, incorporating user_service logic."""

    def __init__(self):
        self.db_path = USER_DB_PATH # Using the centralized path
        self._db_initialized = False
        self._init_database()

    def _connect(self) -> sqlite3.Connection:
        """Internal connection helper (replaces _get_connection from user_service.py)."""
        if not self._db_initialized:
            self._init_database()
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_database(self):
        """
        Initialize user database, including user_meta and seeding dummy users.
        (Integrates _init_db and DUMMY_USERS logic from user_service.py)
        """
        if self._db_initialized:
            return

        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Initializing user database at {self.db_path}")
            with sqlite3.connect(self.db_path) as conn:
                # 1. Users table (from user_manager.py/user_service.py)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        user_id TEXT PRIMARY KEY,
                        username TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        role TEXT NOT NULL,
                        department TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

                # 2. User metadata table (from user_service.py)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS user_meta (
                        user_id TEXT,
                        meta_key TEXT,
                        meta_value TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_id, meta_key),
                        FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                    )
                """)

                # Seed dummy users
                for user_data in DUMMY_USERS:
                    # Use bcrypt for password hashing
                    hashed_password = bcrypt.hashpw(
                        user_data["password"].encode('utf-8'), bcrypt.gensalt()
                    ).decode('utf-8')

                    try:
                        conn.execute("""
                            INSERT INTO users (user_id, username, password, role, department)
                            VALUES (?, ?, ?, ?, ?)
                        """, (
                            user_data["user_id"],
                            user_data["username"],
                            hashed_password,
                            user_data["role"],
                            user_data["department"]
                        ))
                    except sqlite3.IntegrityError:
                        # User already exists
                        pass

                conn.commit()
                self._db_initialized = True
                logger.info("User database initialized and seeded.")

        except Exception as e:
            logger.error(f"Error initializing user database: {e}")
            raise


    # ---------------------------------------------------------
    # USER CRUD (Functions kept same name/signature as IUserManager)
    # ---------------------------------------------------------

    async def _row_to_user_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite Row to user dictionary (Internal helper)."""
        user_dict = dict(row)
        user_id = user_dict['user_id']

        # Fetch metadata and merge
        with self._connect() as conn:
            meta_rows = conn.execute(
                "SELECT meta_key, meta_value FROM user_meta WHERE user_id = ?",
                (user_id,)
            ).fetchall()

        metadata = {}
        for key, value in meta_rows:
            try:
                # Attempt to deserialize JSON metadata
                metadata[key] = json.loads(value)
            except json.JSONDecodeError:
                metadata[key] = value

        # Remove password field from dictionary before returning
        user_dict.pop('password', None)
        user_dict['metadata'] = metadata
        return user_dict


    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID. (Sync logic from user_service.py wrapped in async)."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row:
                return await self._row_to_user_dict(row)
            return None


    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user by username. (Sync logic from user_service.py wrapped in async)."""
        with self._connect() as conn:
            cursor = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
            row = cursor.fetchone()
            if row:
                return await self._row_to_user_dict(row)
            return None


    async def create_user(self, username: str, password: str, role: str, department: Optional[str] = None, profile: Optional[Dict[str, Any]] = None) -> str:
        """Create a new user. (Sync logic from user_service.py wrapped in async)."""
        user_id = f"u_{uuid.uuid4().hex[:8]}"
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        with self._connect() as conn:
            try:
                conn.execute("""
                    INSERT INTO users (user_id, username, password, role, department)
                    VALUES (?, ?, ?, ?, ?)
                """, (user_id, username, hashed_password, role, department))
                conn.commit()
                logger.info(f"User created: {username} ({user_id})")

                # If profile/metadata is provided, store it
                if profile:
                    for key, value in profile.items():
                        # Use the set_user_metadata logic
                        await self.set_user_metadata(user_id, key, value)

                return user_id
            except sqlite3.IntegrityError:
                logger.warning(f"Failed to create user: Username '{username}' already exists.")
                raise ValueError("Username already exists")
            except Exception as e:
                logger.error(f"Error creating user: {e}")
                raise


    async def update_user(self, user_id: str, **kwargs: Any) -> bool:
        """Update user fields. (Sync logic from user_service.py wrapped in async)."""
        set_clauses = []
        params = []

        # Hash password if provided
        if 'password' in kwargs:
            kwargs['password'] = bcrypt.hashpw(kwargs['password'].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

        for key, value in kwargs.items():
            if key in ["username", "password", "role", "department"]:
                set_clauses.append(f"{key} = ?")
                params.append(value)

        if not set_clauses:
            return True # Nothing to update

        params.append(datetime.utcnow().isoformat())
        params.append(user_id)

        update_query = f"""
            UPDATE users
            SET {', '.join(set_clauses)}, updated_at = ?
            WHERE user_id = ?
        """

        with self._connect() as conn:
            cursor = conn.execute(update_query, tuple(params))
            if cursor.rowcount == 0:
                return False
            conn.commit()

        logger.info(f"User updated: {user_id}")
        return True


    async def delete_user(self, user_id: str) -> bool:
        """Delete user. (Sync logic from user_service.py wrapped in async)."""
        with self._connect() as conn:
            # Foreign key cascade should handle user_meta deletion, but explicit is safer
            conn.execute("DELETE FROM user_meta WHERE user_id = ?", (user_id,))
            cursor = conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
            conn.commit()

            if cursor.rowcount == 0:
                return False

        logger.info(f"User deleted: {user_id}")
        return True

    # ---------------------------------------------------------
    # METADATA MANAGEMENT (Functions kept same name/signature as IUserManager)
    # ---------------------------------------------------------

    async def get_user_metadata(self, user_id: str, key: str) -> Any:
        """Get user metadata value. (Sync logic from user_service.py wrapped in async)."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT meta_value FROM user_meta WHERE user_id = ? AND meta_key = ?",
                (user_id, key)
            )
            row = cursor.fetchone()

            if row:
                try:
                    # Attempt to deserialize JSON metadata
                    return json.loads(row[0])
                except json.JSONDecodeError:
                    return row[0]
            return None


    async def set_user_metadata(self, user_id: str, key: str, value: Any) -> bool:
        """Set user metadata value. (Sync logic from user_service.py wrapped in async)."""
        # Serialize value if it's not a string
        value_string = json.dumps(value) if not isinstance(value, str) else value

        with self._connect() as conn:
            try:
                conn.execute("""
                    INSERT INTO user_meta (user_id, meta_key, meta_value, updated_at)
                    VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                    ON CONFLICT(user_id, meta_key) 
                    DO UPDATE SET meta_value = ?, updated_at = CURRENT_TIMESTAMP
                """, (user_id, key, value_string, value_string))

                conn.commit()
                logger.info(f"Set meta {key}={value} for user {user_id}")
                return True
            except Exception as e:
                conn.rollback()
                logger.exception(f"Error setting user meta: {e}")
                return False