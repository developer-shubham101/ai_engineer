"""Role-Based Access Control implementation."""

from typing import Dict, Any, List
import logging

from .interfaces import IRBACManager
from ..config.constants import ROLE_LEVELS, SENSITIVITY_LEVELS

logger = logging.getLogger(__name__)


class RBACManager(IRBACManager):
    """Role-Based Access Control implementation."""
    
    async def check_permission(self, user: Dict[str, Any], resource: str, action: str) -> bool:
        """Check if user has permission for action on resource."""
        user_role = user.get("role", "Guest")
        user_level = ROLE_LEVELS.get(user_role, 0)
        
        # SuperAdmin has access to everything
        if user_role == "SuperAdmin":
            return True
        
        # Define permission rules
        permission_rules = {
            "documents": {
                "read": 0,  # Everyone can read (filtered by document sensitivity)
                "create": 1,  # Employee+ can create
                "update": 1,  # Employee+ can update (with restrictions)
                "delete": 3   # Manager+ can delete
            },
            "users": {
                "read": 2,    # HR+ can read users
                "create": 3,  # Manager+ can create users
                "update": 2,  # HR+ can update users
                "delete": 4   # SuperAdmin only
            },
            "training": {
                "read": 3,    # Manager+ can view training
                "create": 4,  # SuperAdmin only
                "update": 4,  # SuperAdmin only
                "delete": 4   # SuperAdmin only
            }
        }
        
        required_level = permission_rules.get(resource, {}).get(action, 4)
        granted = user_level >= required_level
        
        await self.log_access_attempt(user, resource, action, granted)
        return granted
    
    async def filter_documents(self, documents: List[Dict[str, Any]], user: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Filter documents based on user permissions."""
        filtered = []
        
        for doc in documents:
            if await self.can_access_document(doc.get("metadata", {}), user):
                filtered.append(doc)
        
        logger.info(f"Filtered {len(documents)} documents to {len(filtered)} for user {user.get('username')}")
        return filtered
    
    async def can_access_document(self, document_metadata: Dict[str, Any], user: Dict[str, Any]) -> bool:
        """Check if user can access specific document."""
        user_role = user.get("role", "Guest")
        user_department = user.get("department", "")
        user_id = user.get("user_id", "")
        user_level = ROLE_LEVELS.get(user_role, 0)
        
        # SuperAdmin has access to everything
        if user_role == "SuperAdmin":
            return True
        
        # Check allowed_roles override
        allowed_roles = document_metadata.get("allowed_roles", [])
        if allowed_roles and user_role in allowed_roles:
            return True
        
        # Check personal documents
        if document_metadata.get("sensitivity") == "personal":
            owner_id = document_metadata.get("owner_id")
            if owner_id == user_id:
                return True
            # HR+ can access personal documents
            if user_level >= 2:
                return True
            return False
        
        # Check sensitivity level
        sensitivity = document_metadata.get("sensitivity", "public_internal")
        required_level = SENSITIVITY_LEVELS.get(sensitivity, 0)
        
        if user_level < required_level:
            return False
        
        # Check department restrictions for department_confidential
        if sensitivity == "department_confidential":
            doc_department = document_metadata.get("department", "")
            if user_level < 2 and user_department != doc_department:
                return False
        
        return True
    
    async def get_user_level(self, user: Dict[str, Any]) -> int:
        """Get user's access level."""
        return ROLE_LEVELS.get(user.get("role", "Guest"), 0)
    
    async def log_access_attempt(self, user: Dict[str, Any], resource: str, action: str, granted: bool) -> None:
        """Log access attempt for audit."""
        log_data = {
            "user_id": user.get("user_id"),
            "username": user.get("username"),
            "role": user.get("role"),
            "department": user.get("department"),
            "resource": resource,
            "action": action,
            "granted": granted
        }
        
        if granted:
            logger.info(f"Access granted: {log_data}")
        else:
            logger.warning(f"Access denied: {log_data}")
    
    def can_create_document_with_sensitivity(self, user: Dict[str, Any], sensitivity: str) -> bool:
        """Check if user can create document with given sensitivity level."""
        user_level = ROLE_LEVELS.get(user.get("role", "Guest"), 0)
        required_level = SENSITIVITY_LEVELS.get(sensitivity, 0)
        
        # Users can only create documents at their level or below
        return user_level >= required_level
    
    def can_update_document(self, user: Dict[str, Any], document_metadata: Dict[str, Any]) -> bool:
        """Check if user can update specific document."""
        user_role = user.get("role", "Guest")
        user_department = user.get("department", "")
        user_id = user.get("user_id", "")
        user_level = ROLE_LEVELS.get(user_role, 0)
        
        # SuperAdmin can update everything
        if user_role == "SuperAdmin":
            return True
        
        # Owner can update their personal documents
        if document_metadata.get("owner_id") == user_id:
            return True
        
        # Check department restrictions
        doc_department = document_metadata.get("department", "")
        if user_level < 2 and user_department != doc_department:
            return False
        
        # Check sensitivity level
        sensitivity = document_metadata.get("sensitivity", "public_internal")
        required_level = SENSITIVITY_LEVELS.get(sensitivity, 0)
        
        return user_level >= required_level