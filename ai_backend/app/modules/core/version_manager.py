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
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    created_by TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    notes TEXT,
                    status TEXT DEFAULT 'active'
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_document_versions_doc_id 
                ON document_versions (document_id)
            """)
    
    async def create_version(self, document_id: str, version: str, content: str, metadata: Dict[str, Any], created_by: str, notes: Optional[str] = None) -> int:
        """Create new document version."""
        metadata_json = json.dumps(metadata)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO document_versions 
                (document_id, version, content, metadata, created_by, notes)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (document_id, version, content, metadata_json, created_by, notes))
            
            version_id = cursor.lastrowid
        
        logger.info(f"Created version {version} for document {document_id}")
        return version_id
    
    async def get_version_history(self, document_id: str) -> List[Dict[str, Any]]:
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
                version["metadata"] = json.loads(version["metadata"])
                versions.append(version)
            
            return versions
    
    async def get_version(self, document_id: str, version: str) -> Optional[Dict[str, Any]]:
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
                version_data["metadata"] = json.loads(version_data["metadata"])
                return version_data
            
            return None
    
    async def get_latest_version(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get latest version of document."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM document_versions 
                WHERE document_id = ? AND status = 'active'
                ORDER BY created_at DESC 
                LIMIT 1
            """, (document_id,))
            
            row = cursor.fetchone()
            if row:
                version_data = dict(row)
                version_data["metadata"] = json.loads(version_data["metadata"])
                return version_data
            
            return None
    
    async def archive_version(self, document_id: str, version: str, archived_by: str) -> bool:
        """Archive a specific version."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE document_versions 
                SET status = 'archived'
                WHERE document_id = ? AND version = ?
            """, (document_id, version))
            
            success = cursor.rowcount > 0
        
        if success:
            logger.info(f"Archived version {version} of document {document_id} by {archived_by}")
        
        return success
    
    async def compare_versions(self, document_id: str, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions of a document."""
        v1 = await self.get_version(document_id, version1)
        v2 = await self.get_version(document_id, version2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        # Simple comparison (in production, use proper diff algorithm)
        comparison = {
            "document_id": document_id,
            "version1": {
                "version": v1["version"],
                "created_at": v1["created_at"],
                "created_by": v1["created_by"],
                "content_length": len(v1["content"])
            },
            "version2": {
                "version": v2["version"],
                "created_at": v2["created_at"],
                "created_by": v2["created_by"],
                "content_length": len(v2["content"])
            },
            "content_changed": v1["content"] != v2["content"],
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