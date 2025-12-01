"""Document management service."""

from typing import Dict, Any, List, Optional
import uuid
from datetime import datetime
import logging

from ..vector_db.interfaces import IVectorStore
from ..auth.interfaces import IRBACManager

logger = logging.getLogger(__name__)


class DocumentManager:
    """Document management service."""
    
    def __init__(self, vector_store: IVectorStore, rbac_manager: IRBACManager):
        self.vector_store = vector_store
        self.rbac_manager = rbac_manager
    
    async def add_document(self, text: str, metadata: Dict[str, Any], user: Dict[str, Any]) -> str:
        """Add new document."""
        # Validate user permissions
        sensitivity = metadata.get("sensitivity", "public_internal")
        if not await self._can_create_document(user, sensitivity):
            raise PermissionError(f"User cannot create document with sensitivity: {sensitivity}")
        
        # Add document metadata
        metadata.update({
            "created_by": user.get("user_id"),
            "created_at": datetime.utcnow().isoformat(),
            "version": "1.0",
            "status": "published"
        })
        
        # Add to vector store
        document_id = await self.vector_store.add_document(text, metadata)
        
        logger.info(f"Document added: {document_id} by user: {user.get('username')}")
        return document_id
    
    async def update_document(self, document_id: str, text: str, metadata: Dict[str, Any], user: Dict[str, Any]) -> bool:
        """Update existing document."""
        # Get existing document
        existing_doc = await self.vector_store.get_document(document_id)
        if not existing_doc:
            raise ValueError(f"Document not found: {document_id}")
        
        # Check update permissions
        if not await self._can_update_document(user, existing_doc.get("metadata", {})):
            raise PermissionError("User cannot update this document")
        
        # Update metadata
        metadata.update({
            "updated_by": user.get("user_id"),
            "updated_at": datetime.utcnow().isoformat()
        })
        
        # Update in vector store
        success = await self.vector_store.update_document(document_id, text, metadata)
        
        if success:
            logger.info(f"Document updated: {document_id} by user: {user.get('username')}")
        
        return success
    
    async def delete_document(self, document_id: str, user: Dict[str, Any]) -> bool:
        """Delete document."""
        # Get existing document
        existing_doc = await self.vector_store.get_document(document_id)
        if not existing_doc:
            raise ValueError(f"Document not found: {document_id}")
        
        # Check delete permissions (requires Manager+ level)
        user_level = await self.rbac_manager.get_user_level(user)
        if user_level < 3:  # Manager level
            raise PermissionError("Only Managers and above can delete documents")
        
        # Delete from vector store
        success = await self.vector_store.delete_document(document_id)
        
        if success:
            logger.info(f"Document deleted: {document_id} by user: {user.get('username')}")
        
        return success
    
    async def list_documents(self, user: Dict[str, Any], filter_metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """List documents accessible to user."""
        # Get all documents matching filter
        documents = await self.vector_store.list_documents(filter_metadata)
        
        # Apply RBAC filtering
        filtered_documents = await self.rbac_manager.filter_documents(documents, user)
        
        return filtered_documents
    
    async def search_documents(self, query: str, user: Dict[str, Any], top_k: int = 10, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search documents accessible to user."""
        filter_metadata = {}
        if category:
            filter_metadata["category"] = category
        
        # Search documents
        documents = await self.vector_store.search_documents(query, top_k * 2, filter_metadata)
        
        # Apply RBAC filtering
        filtered_documents = await self.rbac_manager.filter_documents(documents, user)
        
        return filtered_documents[:top_k]
    
    async def _can_create_document(self, user: Dict[str, Any], sensitivity: str) -> bool:
        """Check if user can create document with given sensitivity."""
        from ..config.constants import ROLE_LEVELS, SENSITIVITY_LEVELS
        
        user_level = ROLE_LEVELS.get(user.get("role", "Guest"), 0)
        required_level = SENSITIVITY_LEVELS.get(sensitivity, 0)
        
        return user_level >= required_level
    
    async def _can_update_document(self, user: Dict[str, Any], document_metadata: Dict[str, Any]) -> bool:
        """Check if user can update document."""
        return await self.rbac_manager.can_access_document(document_metadata, user)