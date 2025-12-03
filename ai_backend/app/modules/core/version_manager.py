"""Document version management service."""

import sqlite3
import json
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from ..config.settings import settings

logger = logging.getLogger(__name__)


class VersionManager:
    """Document version management service."""
    
    def __init__(self):
        self.db_path = settings.DATABASE_DIR / "document_versions.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize version database."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
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
            
            # Create indexes for better query performance
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_id 
                ON document_versions (document_id)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_version_status 
                ON document_versions (status)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at 
                ON document_versions (created_at DESC)
            """)
    
    async def create_version(
        self,
        document_id: str,
        version: str,
        source_name: str,
        chunk_ids: List[str],
        created_by: Optional[str] = None,
        parent_version: Optional[str] = None,
        status: str = "published",
        version_notes: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Create new document version.
        
        Args:
            document_id: Unique document identifier
            version: Version string (e.g., "1.0", "2.0")
            source_name: Original source file name
            chunk_ids: List of chunk IDs associated with this version
            created_by: User who created this version
            parent_version: Previous version this is based on
            status: Version status (published, draft, pending_approval, archived)
            version_notes: Optional version notes
            metadata: Optional metadata dictionary
            
        Returns:
            Version ID of the created record
        """
        created_at = datetime.utcnow().isoformat() + "Z"
        chunk_ids_json = json.dumps(chunk_ids)
        metadata_json = json.dumps(metadata) if metadata else None
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO document_versions 
                    (document_id, version, source_name, chunk_ids, created_at, created_by,
                     parent_version, status, version_notes, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (document_id, version, source_name, chunk_ids_json, created_at, created_by,
                      parent_version, status, version_notes, metadata_json))
                
                version_id = cursor.lastrowid
            
            logger.info(f"Created version {version} for document {document_id}")
            return version_id
        except sqlite3.IntegrityError as e:
            logger.error(f"Version already exists: document_id={document_id} version={version}")
            raise ValueError(f"Version {version} already exists for document {document_id}") from e
    
    async def create_version_record(
        self,
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
        """Create a new version record in the database.
        
        Alternative to create_version() that returns boolean instead of version ID.
        
        Args:
            document_id: Unique document identifier
            version: Version string (e.g., "1.0", "2.0")
            source_name: Original source file name
            chunk_ids: List of chunk IDs associated with this version
            created_by: User who created this version
            parent_version: Previous version this is based on
            status: Version status (published, draft, pending_approval, archived)
            version_notes: Optional version notes
            metadata: Optional metadata dictionary
            
        Returns:
            True if successful, False otherwise
        """
        created_at = datetime.utcnow().isoformat() + "Z"
        chunk_ids_json = json.dumps(chunk_ids)
        metadata_json = json.dumps(metadata) if metadata else None
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    INSERT INTO document_versions 
                    (document_id, version, source_name, chunk_ids, created_at, created_by,
                     parent_version, status, version_notes, metadata_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (document_id, version, source_name, chunk_ids_json, created_at, created_by,
                      parent_version, status, version_notes, metadata_json))
            
            logger.info(f"Created version record: document_id={document_id} version={version}")
            return True
        except sqlite3.IntegrityError as e:
            logger.error(f"Version already exists: document_id={document_id} version={version} error={e}")
            return False
        except Exception as e:
            logger.exception(f"Failed to create version record: {e}")
            return False
    
    def get_version_history(self, document_id: str) -> List[Dict[str, Any]]:
        """Get version history for document."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM document_versions 
                WHERE document_id = ? 
                ORDER BY created_at DESC
            """, (document_id,))
            
            versions = []
            for row in cursor.fetchall():
                version = dict(row)
                version["chunk_ids"] = json.loads(version["chunk_ids"])
                version["metadata"] = json.loads(version["metadata"]) if version["metadata"] else None
                versions.append(version)
            
            return versions
    
    def get_version(self, document_id: str, version: str) -> Optional[Dict[str, Any]]:
        """Get specific version of document."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM document_versions 
                WHERE document_id = ? AND version = ?
            """, (document_id, version))
            
            row = cursor.fetchone()
            if row:
                version_data = dict(row)
                version_data["chunk_ids"] = json.loads(version_data["chunk_ids"])
                version_data["metadata"] = json.loads(version_data["metadata"]) if version_data["metadata"] else None
                return version_data
            
            return None

    def get_latest_version(self, document_id: str, status: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get latest version of document.
        
        Args:
            document_id: Document identifier
            status: Optional status filter (if None, gets latest regardless of status)
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if status:
                cursor = conn.execute("""
                    SELECT * FROM document_versions 
                    WHERE document_id = ? AND status = ?
                    ORDER BY created_at DESC 
                    LIMIT 1
                """, (document_id, status))
            else:
                cursor = conn.execute("""
                    SELECT * FROM document_versions 
                    WHERE document_id = ?
                    ORDER BY created_at DESC 
                    LIMIT 1
                """, (document_id,))
            
            row = cursor.fetchone()
            if row:
                version_data = dict(row)
                version_data["chunk_ids"] = json.loads(version_data["chunk_ids"])
                version_data["metadata"] = json.loads(version_data["metadata"]) if version_data["metadata"] else None
                return version_data
            
            return None
    
    async def update_version_status(self, document_id: str, version: str, status: str) -> bool:
        """Update the status of a version.
        
        Args:
            document_id: Document ID
            version: Version number
            status: New status (draft, pending_approval, published, archived)
        
        Returns:
            True if successful, False otherwise
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE document_versions 
                SET status = ?
                WHERE document_id = ? AND version = ?
            """, (status, document_id, version))
            
            success = cursor.rowcount > 0
        
        if success:
            logger.info(f"Updated version status: document_id={document_id} version={version} status={status}")
        else:
            logger.warning(f"No version found to update: document_id={document_id} version={version}")
        
        return success
    
    async def archive_version(self, document_id: str, version: str, archived_by: str) -> bool:
        """Archive a specific version (convenience method)."""
        return await self.update_version_status(document_id, version, "archived")
    
    async def compare_versions(self, document_id: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions of a document."""
        v1 = self.get_version(document_id, version1)
        v2 = self.get_version(document_id, version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        # Simple comparison (in production, use proper diff algorithm)
        comparison = {
            "document_id": document_id,
            "version1": {
                "version": v1["version"],
                "created_at": v1["created_at"],
                "created_by": v1["created_by"],
                "chunk_count": len(v1["chunk_ids"])
            },
            "version2": {
                "version": v2["version"],
                "created_at": v2["created_at"],
                "created_by": v2["created_by"],
                "chunk_count": len(v2["chunk_ids"])
            },
            "chunk_ids_changed": v1["chunk_ids"] != v2["chunk_ids"],
            "metadata_changed": v1["metadata"] != v2["metadata"]
        }
        
        return comparison
    
    async def cleanup_old_versions(self, document_id: str, keep_count: int = 10) -> int:
        """Clean up old versions, keeping only the most recent ones."""
        with sqlite3.connect(self.db_path) as conn:
            # Get versions to delete (older than keep_count)
            cursor = conn.execute("""
                SELECT id FROM document_versions 
                WHERE document_id = ? 
                ORDER BY created_at DESC 
                LIMIT -1 OFFSET ?
            """, (document_id, keep_count))
            
            version_ids = [row[0] for row in cursor.fetchall()]
            
            if version_ids:
                placeholders = ",".join("?" * len(version_ids))
                conn.execute(f"""
                    DELETE FROM document_versions 
                    WHERE id IN ({placeholders})
                """, version_ids)
                
                deleted_count = len(version_ids)
                logger.info(f"Cleaned up {deleted_count} old versions for document {document_id}")
                return deleted_count
            
            return 0
    
    def get_documents_by_status(self, status: str) -> List[Dict[str, Any]]:
        """Get all versions with a specific status.
        
        Args:
            status: Status to filter by (draft, pending_approval, published, archived)
        
        Returns:
            List of version records
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM document_versions
                WHERE status = ?
                ORDER BY created_at DESC
            """, (status,))
            
            versions = []
            for row in cursor.fetchall():
                version = dict(row)
                version["chunk_ids"] = json.loads(version["chunk_ids"])
                version["metadata"] = json.loads(version["metadata"]) if version["metadata"] else None
                versions.append(version)
            
            return versions
    
    def list_all_documents(self, latest_only: bool = True) -> List[Dict[str, Any]]:
        """List all documents (optionally only latest versions).
        
        Args:
            latest_only: If True, return only the latest version of each document
        
        Returns:
            List of document summaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if latest_only:
                # Get only the latest version of each document
                cursor = conn.execute("""
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
                cursor = conn.execute("""
                    SELECT document_id, version, source_name, created_at, created_by, status
                    FROM document_versions
                    ORDER BY created_at DESC
                """)
            
            documents = []
            for row in cursor.fetchall():
                documents.append(dict(row))
            
            return documents
    
    def generate_next_version(self, document_id: str) -> str:
        """Generate the next version number for a document.
        Uses semantic versioning (1.0, 2.0, 3.0, etc.)
        
        Args:
            document_id: Document identifier
        
        Returns:
            Next version string (e.g., "1.0" for new doc, "2.0" for first update)
        """
        latest = self.get_latest_version(document_id)
        if not latest:
            return "1.0"
        
        try:
            # Parse current version (e.g., "1.0" -> 1)
            current_version = latest["version"]
            major = int(float(current_version))
            next_major = major + 1
            return f"{next_major}.0"
        except Exception as e:
            logger.warning(f"Failed to parse version {latest.get('version')}, defaulting to 1.0: {e}")
            return "1.0"
