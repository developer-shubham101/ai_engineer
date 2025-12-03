# app/dependencies.py
"""
Dependency injection providers for FastAPI.
"""
from typing import Optional, List, Dict, Any
from fastapi import Depends, HTTPException, status, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from app.modules.integration import get_container
from app.modules.config.constants import HTTP_MESSAGES

logger = logging.getLogger(__name__)

# Security scheme for Bearer token
security = HTTPBearer(auto_error=False, scheme_name="Bearer", description="Enter your JWT token")


def get_rag_service():
    """
    Dependency provider that returns the RAG orchestrator.
    This allows for easy mocking in tests by overriding this dependency.
    """
    container = get_container()
    container.initialize()
    return container.get_rag_orchestrator()


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Dict[str, Any]:
    """
    Dependency to get the current authenticated user from Bearer token.
    Raises 401 if no valid token is provided.
    
    Authentication:
    1. Authorization: Bearer <token> (JWT token)
    
    Returns:
        User dict with user_id, username, role, department
    """
    # Try Bearer token
    if credentials:
        token = credentials.credentials
        container = get_container()
        container.initialize()
        authenticator = container.get_authenticator()
        user = await authenticator.verify_token(token)
        if user:
            return user
    
    # No valid authentication found
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=HTTP_MESSAGES["UNAUTHORIZED"],
        headers={"WWW-Authenticate": "Bearer"},
    )


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Optional[Dict[str, Any]]:
    """
    Dependency to optionally get the current authenticated user from Bearer token.
    Returns None if no valid token is provided (treats as Guest user).
    
    Authentication:
    1. Authorization: Bearer <token> (JWT token)
    
    Returns:
        User dict with user_id, username, role, department, or None for Guest
    """
    # Try Bearer token
    if credentials:
        token = credentials.credentials
        container = get_container()
        container.initialize()
        authenticator = container.get_authenticator()
        user = await authenticator.verify_token(token)
        if user:
            return user
    
    # No authentication provided - treat as Guest
    return None


def require_roles(allowed_roles: List[str]):
    """
    Dependency factory for role-based access control.
    
    Usage:
        @router.post("/admin-only", dependencies=[Depends(require_roles(["SuperAdmin"]))])
        
    Args:
        allowed_roles: List of roles that are allowed to access the endpoint
        
    Returns:
        Dependency function that checks if user has required role
    """
    async def check_role(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_role = current_user.get("role")
        
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"{HTTP_MESSAGES['FORBIDDEN']}. Required roles: {', '.join(allowed_roles)}"
            )
        
        return current_user
    
    return check_role
