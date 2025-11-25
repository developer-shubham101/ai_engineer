# app/services/user_service.py
"""
User management service with SQLite database.
Handles user authentication, password hashing, and user CRUD operations.
"""
import sqlite3
import logging
import bcrypt
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Database path
from app.services.utility import DATA_DIR
USER_DB_PATH = DATA_DIR / "users.db"

# Dummy users to seed on initialization
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
        "department": "Engineering"
    },
    {
        "username": "guest",
        "password": "guest123",
        "user_id": "u_guest_1",
        "role": "Guest",
        "department": "General"
    },
]


def get_password_hash(password: str) -> str:
    """Hash a password using bcrypt."""
    # Convert password to bytes and hash it
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    # Return as string for storage
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    password_bytes = plain_password.encode('utf-8')
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def _get_connection() -> sqlite3.Connection:
    """Get a database connection."""
    conn = sqlite3.Connection(str(USER_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_user_db(reset_on_start: bool = False) -> None:
    """
    Initialize the user database and seed dummy users.
    
    Args:
        reset_on_start: If True, drop and recreate the table (for development)
    """
    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    conn = _get_connection()
    cursor = conn.cursor()
    
    try:
        if reset_on_start:
            cursor.execute("DROP TABLE IF EXISTS users")
            logger.info("Dropped existing users table")
        
        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                user_id TEXT UNIQUE NOT NULL,
                role TEXT NOT NULL,
                department TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Create user_meta table for dynamic profile fields
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS user_meta (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                meta_key TEXT NOT NULL,
                meta_value TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(user_id, meta_key),
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # Check if we need to seed
        cursor.execute("SELECT COUNT(*) as count FROM users")
        count = cursor.fetchone()["count"]
        
        if count == 0:
            logger.info("Seeding dummy users...")
            for user in DUMMY_USERS:
                password_hash = get_password_hash(user["password"])
                cursor.execute("""
                    INSERT INTO users (username, password_hash, user_id, role, department)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    user["username"],
                    password_hash,
                    user["user_id"],
                    user["role"],
                    user["department"]
                ))
            logger.info(f"Seeded {len(DUMMY_USERS)} dummy users")
            
            # Seed basic profiles for dummy users
            logger.info("Seeding user profiles...")
            profiles = {
                "u_admin_1": {"name": "Admin User", "gender": "Other", "location": "HQ"},
                "u_hr_1": {"name": "HR Manager", "gender": "Female", "location": "New York"},
                "u_mgr_1": {"name": "Engineering Manager", "gender": "Male", "location": "San Francisco"},
                "u_emp_1": {"name": "John Employee", "gender": "Male", "location": "Austin"},
            }
            
            for user_id, profile in profiles.items():
                for key, value in profile.items():
                    cursor.execute("""
                        INSERT INTO user_meta (user_id, meta_key, meta_value)
                        VALUES (?, ?, ?)
                    """, (user_id, key, value))
            logger.info(f"Seeded profiles for {len(profiles)} users")
        else:
            logger.info(f"User database already contains {count} users")
        
        conn.commit()
        logger.info("User database initialized successfully")
        
    except Exception as e:
        conn.rollback()
        logger.exception(f"Error initializing user database: {e}")
        raise
    finally:
        conn.close()


def get_user_by_username(username: str) -> Optional[Dict[str, Any]]:
    """
    Retrieve a user by username.
    
    Returns:
        User dict with keys: id, username, password_hash, user_id, role, department, created_at
        None if user not found
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT id, username, password_hash, user_id, role, department, created_at
            FROM users
            WHERE username = ?
        """, (username,))
        
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
        
    finally:
        conn.close()


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    """
    Authenticate a user with username and password.
    
    Returns:
        User dict (without password_hash) if authentication successful
        None if authentication failed
    """
    user = get_user_by_username(username)
    
    if not user:
        logger.warning(f"Authentication failed: user '{username}' not found")
        return None
    
    if not verify_password(password, user["password_hash"]):
        logger.warning(f"Authentication failed: invalid password for user '{username}'")
        return None
    
    # Remove password hash from returned user data
    user_data = {
        "user_id": user["user_id"],
        "username": user["username"],
        "role": user["role"],
        "department": user["department"]
    }
    
    logger.info(f"User '{username}' authenticated successfully")
    return user_data


def create_user(username: str, password: str, user_id: str, role: str, department: str) -> bool:
    """
    Create a new user.
    
    Returns:
        True if user created successfully
        False if user already exists or error occurred
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    try:
        password_hash = get_password_hash(password)
        cursor.execute("""
            INSERT INTO users (username, password_hash, user_id, role, department)
            VALUES (?, ?, ?, ?, ?)
        """, (username, password_hash, user_id, role, department))
        
        conn.commit()
        logger.info(f"User '{username}' created successfully")
        return True
        
    except sqlite3.IntegrityError as e:
        logger.warning(f"Failed to create user '{username}': {e}")
        return False
    except Exception as e:
        conn.rollback()
        logger.exception(f"Error creating user '{username}': {e}")
        return False
    finally:
        conn.close()


# ============================================================================
# User Meta Functions
# ============================================================================

def get_user_meta(user_id: str, key: str) -> Optional[str]:
    """
    Get a single user meta value.
    
    Args:
        user_id: User ID
        key: Meta key
        
    Returns:
        Meta value if found, None otherwise
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT meta_value FROM user_meta
            WHERE user_id = ? AND meta_key = ?
        """, (user_id, key))
        
        row = cursor.fetchone()
        return row["meta_value"] if row else None
        
    finally:
        conn.close()


def get_all_user_meta(user_id: str) -> Dict[str, str]:
    """
    Get all user meta as a dictionary.
    
    Args:
        user_id: User ID
        
    Returns:
        Dictionary of meta key-value pairs
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT meta_key, meta_value FROM user_meta
            WHERE user_id = ?
        """, (user_id,))
        
        rows = cursor.fetchall()
        return {row["meta_key"]: row["meta_value"] for row in rows}
        
    finally:
        conn.close()


def set_user_meta(user_id: str, key: str, value: str) -> bool:
    """
    Set or update a user meta value.
    
    Args:
        user_id: User ID
        key: Meta key
        value: Meta value
        
    Returns:
        True if successful, False otherwise
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            INSERT INTO user_meta (user_id, meta_key, meta_value, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id, meta_key) 
            DO UPDATE SET meta_value = ?, updated_at = CURRENT_TIMESTAMP
        """, (user_id, key, value, value))
        
        conn.commit()
        logger.info(f"Set meta {key}={value} for user {user_id}")
        return True
        
    except Exception as e:
        conn.rollback()
        logger.exception(f"Error setting user meta: {e}")
        return False
    finally:
        conn.close()


def delete_user_meta(user_id: str, key: str) -> bool:
    """
    Delete a user meta value.
    
    Args:
        user_id: User ID
        key: Meta key
        
    Returns:
        True if successful, False otherwise
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            DELETE FROM user_meta
            WHERE user_id = ? AND meta_key = ?
        """, (user_id, key))
        
        conn.commit()
        logger.info(f"Deleted meta {key} for user {user_id}")
        return True
        
    except Exception as e:
        conn.rollback()
        logger.exception(f"Error deleting user meta: {e}")
        return False
    finally:
        conn.close()
