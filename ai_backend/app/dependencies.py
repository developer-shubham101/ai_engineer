# app/dependencies.py
"""
Dependency injection providers for FastAPI.
"""
from typing import Optional, List
from fastapi import Depends, HTTPException, status, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

from app.services import rag_local_service
from app.services.auth import verify_token, get_user_from_api_key

logger = logging.getLogger(__name__)

# Security scheme for Bearer token
security = HTTPBearer(auto_error=False)


def get_rag_service():
    """
    Dependency provider that returns the RAG service module.
    This allows for easy mocking in tests by overriding this dependency.
    """
    return rag_local_service


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None)
) -> dict:
    """
    Get the current authenticated user from Bearer token or API key.
    
    Priority:
    1. Bearer token (Authorization header)
    2. X-API-Key header (legacy support)
    3. Raise 401 if neither provided
    
    Returns:
        User dict with user_id, role, department
    """
    # Try Bearer token first
    if credentials and credentials.credentials:
        token_data = verify_token(credentials.credentials)
        if token_data:
            return {
                "user_id": token_data.get("user_id"),
                "username": token_data.get("username"),
                "role": token_data.get("role"),
                "department": token_data.get("department")
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    # Try legacy API key
    if x_api_key:
        user = get_user_from_api_key(x_api_key)
        if user:
            return user
    
    # No authentication provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated",
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    x_api_key: Optional[str] = Header(None)
) -> Optional[dict]:
    """
    Get the current user if authenticated, otherwise return None (Guest).
    Used for endpoints that allow both authenticated and unauthenticated access.
    
    Returns:
        User dict if authenticated
        None if not authenticated (Guest user)
    """
    # Try Bearer token
    if credentials and credentials.credentials:
        token_data = verify_token(credentials.credentials)
        if token_data:
            return {
                "user_id": token_data.get("user_id"),
                "username": token_data.get("username"),
                "role": token_data.get("role"),
                "department": token_data.get("department")
            }
    
    # Try legacy API key
    if x_api_key:
        user = get_user_from_api_key(x_api_key)
        if user:
            return user
    
    # Return None for Guest users
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
    def check_role(current_user: dict = Depends(get_current_user)):
        user_role = current_user.get("role")
        
        if user_role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join(allowed_roles)}"
            )
        
        return current_user
    
    return check_role
