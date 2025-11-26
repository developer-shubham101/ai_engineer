# app/services/version_tracking.py
"""
Document version tracking service.

Manages document version history using SQLite database.
Tracks version metadata, status, and relationships between versions.
"""
from __future__ import annotations

import sqlite3
import logging
import json
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

# Database path
from app.services.utility import get_data_path

VERSION_DB_PATH = get_data_path("document_versions.db")


def _get_connection() -> sqlite3.Connection:
    """Get a database connection."""
    conn = sqlite3.connect(str(VERSION_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_version_db(reset_on_start: bool = False) -> None:
    """
    Initialize the version tracking database.
    
    Args:
        reset_on_start: If True, drop and recreate the table (for development)
    """
    conn = _get_connection()
    cursor = conn.cursor()
    
    if reset_on_start:
        logger.warning("Resetting document_versions table (reset_on_start=True)")
        cursor.execute("DROP TABLE IF EXISTS document_versions")
    
    # Create table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS document_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            version TEXT NOT NULL,
            source_name TEXT NOT NULL,
            chunk_ids TEXT NOT NULL,
            created_at TEXT NOT NULL,
            created_by TEXT,
            parent_version TEXT,
            status TEXT DEFAULT 'published',
            version_notes TEXT,
            metadata_json TEXT,
            UNIQUE(document_id, version)
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_document_id ON document_versions(document_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_version_status ON document_versions(status)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON document_versions(created_at DESC)")
    
    conn.commit()
    conn.close()
    logger.info("Version tracking database initialized at %s", VERSION_DB_PATH)


def create_version_record(
    document_id: str,
    version: str,
    source_name: str,
    chunk_ids: List[str],
    created_by: Optional[str] = None,
    parent_version: Optional[str] = None,
    status: str = "published",
    version_notes: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Create a new version record in the database.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        
        created_at = datetime.utcnow().isoformat() + "Z"
        chunk_ids_json = json.dumps(chunk_ids)
        metadata_json = json.dumps(metadata) if metadata else None
        
        cursor.execute("""
            INSERT INTO document_versions 
            (document_id, version, source_name, chunk_ids, created_at, created_by, 
             parent_version, status, version_notes, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (document_id, version, source_name, chunk_ids_json, created_at, created_by,
              parent_version, status, version_notes, metadata_json))
        
        conn.commit()
        conn.close()
        logger.info("Created version record: document_id=%s version=%s", document_id, version)
        return True
    except sqlite3.IntegrityError as e:
        logger.error("Version already exists: document_id=%s version=%s error=%s", 
                    document_id, version, e)
        return False
    except Exception as e:
        logger.exception("Failed to create version record: %s", e)
        return False


def get_version_history(document_id: str) -> List[Dict[str, Any]]:
    """
    Get all versions for a document, ordered by created_at DESC.
    
    Returns:
        List of version records
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, document_id, version, source_name, chunk_ids, created_at,
                   created_by, parent_version, status, version_notes, metadata_json
            FROM document_versions
            WHERE document_id = ?
            ORDER BY created_at DESC
        """, (document_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        versions = []
        for row in rows:
            versions.append({
                "id": row["id"],
                "document_id": row["document_id"],
                "version": row["version"],
                "source_name": row["source_name"],
                "chunk_ids": json.loads(row["chunk_ids"]),
                "created_at": row["created_at"],
                "created_by": row["created_by"],
                "parent_version": row["parent_version"],
                "status": row["status"],
                "version_notes": row["version_notes"],
                "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else None
            })
        
        return versions
    except Exception as e:
        logger.exception("Failed to get version history: %s", e)
        return []


def get_version(document_id: str, version: str) -> Optional[Dict[str, Any]]:
    """
    Get a specific version of a document.
    
    Returns:
        Version record or None if not found
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, document_id, version, source_name, chunk_ids, created_at,
                   created_by, parent_version, status, version_notes, metadata_json
            FROM document_versions
            WHERE document_id = ? AND version = ?
        """, (document_id, version))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            "id": row["id"],
            "document_id": row["document_id"],
            "version": row["version"],
            "source_name": row["source_name"],
            "chunk_ids": json.loads(row["chunk_ids"]),
            "created_at": row["created_at"],
            "created_by": row["created_by"],
            "parent_version": row["parent_version"],
            "status": row["status"],
            "version_notes": row["version_notes"],
            "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else None
        }
    except Exception as e:
        logger.exception("Failed to get version: %s", e)
        return None


def get_latest_version(document_id: str) -> Optional[Dict[str, Any]]:
    """
    Get the most recent version of a document.
    
    Returns:
        Latest version record or None if not found
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, document_id, version, source_name, chunk_ids, created_at,
                   created_by, parent_version, status, version_notes, metadata_json
            FROM document_versions
            WHERE document_id = ?
            ORDER BY created_at DESC
            LIMIT 1
        """, (document_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return {
            "id": row["id"],
            "document_id": row["document_id"],
            "version": row["version"],
            "source_name": row["source_name"],
            "chunk_ids": json.loads(row["chunk_ids"]),
            "created_at": row["created_at"],
            "created_by": row["created_by"],
            "parent_version": row["parent_version"],
            "status": row["status"],
            "version_notes": row["version_notes"],
            "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else None
        }
    except Exception as e:
        logger.exception("Failed to get latest version: %s", e)
        return None


def update_version_status(document_id: str, version: str, status: str) -> bool:
    """
    Update the status of a version.
    
    Args:
        document_id: Document ID
        version: Version number
        status: New status (draft, pending_approval, published, archived)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE document_versions
            SET status = ?
            WHERE document_id = ? AND version = ?
        """, (status, document_id, version))
        
        conn.commit()
        affected = cursor.rowcount
        conn.close()
        
        if affected > 0:
            logger.info("Updated version status: document_id=%s version=%s status=%s",
                       document_id, version, status)
            return True
        else:
            logger.warning("No version found to update: document_id=%s version=%s",
                          document_id, version)
            return False
    except Exception as e:
        logger.exception("Failed to update version status: %s", e)
        return False


def get_documents_by_status(status: str) -> List[Dict[str, Any]]:
    """
    Get all versions with a specific status.
    
    Returns:
        List of version records
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, document_id, version, source_name, chunk_ids, created_at,
                   created_by, parent_version, status, version_notes, metadata_json
            FROM document_versions
            WHERE status = ?
            ORDER BY created_at DESC
        """, (status,))
        
        rows = cursor.fetchall()
        conn.close()
        
        versions = []
        for row in rows:
            versions.append({
                "id": row["id"],
                "document_id": row["document_id"],
                "version": row["version"],
                "source_name": row["source_name"],
                "chunk_ids": json.loads(row["chunk_ids"]),
                "created_at": row["created_at"],
                "created_by": row["created_by"],
                "parent_version": row["parent_version"],
                "status": row["status"],
                "version_notes": row["version_notes"],
                "metadata": json.loads(row["metadata_json"]) if row["metadata_json"] else None
            })
        
        return versions
    except Exception as e:
        logger.exception("Failed to get documents by status: %s", e)
        return []


def list_all_documents(latest_only: bool = True) -> List[Dict[str, Any]]:
    """
    List all documents (optionally only latest versions).
    
    Returns:
        List of document summaries
    """
    try:
        conn = _get_connection()
        cursor = conn.cursor()
        
        if latest_only:
            # Get only the latest version of each document
            cursor.execute("""
                SELECT document_id, version, source_name, created_at, created_by, status
                FROM document_versions dv1
                WHERE created_at = (
                    SELECT MAX(created_at)
                    FROM document_versions dv2
                    WHERE dv2.document_id = dv1.document_id
                )
                ORDER BY created_at DESC
            """)
        else:
            cursor.execute("""
                SELECT document_id, version, source_name, created_at, created_by, status
                FROM document_versions
                ORDER BY created_at DESC
            """)
        
        rows = cursor.fetchall()
        conn.close()
        
        documents = []
        for row in rows:
            documents.append({
                "document_id": row["document_id"],
                "version": row["version"],
                "source_name": row["source_name"],
                "created_at": row["created_at"],
                "created_by": row["created_by"],
                "status": row["status"]
            })
        
        return documents
    except Exception as e:
        logger.exception("Failed to list documents: %s", e)
        return []


def generate_next_version(document_id: str) -> str:
    """
    Generate the next version number for a document.
    Uses semantic versioning (1.0, 2.0, 3.0, etc.)
    
    Returns:
        Next version string (e.g., "1.0" for new doc, "2.0" for first update)
    """
    latest = get_latest_version(document_id)
    if not latest:
        return "1.0"
    
    try:
        # Parse current version (e.g., "1.0" -> 1)
        current_version = latest["version"]
        major = int(float(current_version))
        next_major = major + 1
        return f"{next_major}.0"
    except Exception as e:
        logger.warning("Failed to parse version %s, defaulting to 1.0: %s", 
                      latest.get("version"), e)
        return "1.0"
