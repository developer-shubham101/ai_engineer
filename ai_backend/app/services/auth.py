# app/services/auth.py
"""
Authentication service with JWT token support.
Handles token generation, verification, and user authentication.
"""
import jwt
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import logging

from app.config import JWT_SECRET_KEY, JWT_ALGORITHM, JWT_EXPIRATION_DAYS

logger = logging.getLogger(__name__)

# Legacy API keys (kept for backward compatibility during transition)
_API_KEYS = {
    # ============================================================
    # 🟦 GENERAL EMPLOYEES (Regular Users)
    # ============================================================
    "key-employee-1": {
        "user_id": "u_emp_1",
        "role": "Employee",
        "department": "Engineering",
    },
    "key-employee-2": {
        "user_id": "u_emp_2",
        "role": "Employee",
        "department": "Finance",
    },
    "key-employee-3": {
        "user_id": "u_emp_3",
        "role": "Employee",
        "department": "HR",
    },

    # ============================================================
    # 🟦 ENGINEERING ROLES
    # ============================================================
    "key-engineer-1": {
        "user_id": "u_eng_1",
        "role": "Engineer",
        "department": "Engineering",
    },
    "key-senior-engineer-1": {
        "user_id": "u_seng_1",
        "role": "SeniorEngineer",
        "department": "Engineering",
    },
    "key-engineering-manager-1": {
        "user_id": "u_engmgr_1",
        "role": "EngineeringManager",
        "department": "Engineering",
    },

    # ============================================================
    # 🟧 MANAGERS
    # ============================================================
    "key-manager-1": {
        "user_id": "u_mgr_1",
        "role": "Manager",
        "department": "Engineering",
    },
    "key-manager-2": {
        "user_id": "u_mgr_2",
        "role": "Manager",
        "department": "Finance",
    },

    # ============================================================
    # 🟪 HR TEAM
    # ============================================================
    "key-hr-1": {
        "user_id": "u_hr_1",
        "role": "HR",
        "department": "HR",
    },
    "key-hr-manager-1": {
        "user_id": "u_hrmgr_1",
        "role": "HRManager",
        "department": "HR",
    },

    # ============================================================
    # 🟥 LEGAL TEAM
    # ============================================================
    "key-legal-1": {
        "user_id": "u_legal_1",
        "role": "Legal",
        "department": "Legal",
    },
    "key-legal-contract-1": {
        "user_id": "u_legal_contract_1",
        "role": "LegalAdvisor",
        "department": "Legal",
    },

    # ============================================================
    # 🟩 FINANCE TEAM
    # ============================================================
    "key-finance-1": {
        "user_id": "u_fin_1",
        "role": "FinanceAssociate",
        "department": "Finance",
    },
    "key-finance-manager-1": {
        "user_id": "u_finmgr_1",
        "role": "FinanceManager",
        "department": "Finance",
    },

    # ============================================================
    # 🟨 IT & SECURITY ROLES
    # ============================================================
    "key-it-1": {
        "user_id": "u_it_1",
        "role": "ITSupport",
        "department": "IT",
    },
    "key-it-admin-1": {
        "user_id": "u_itadmin_1",
        "role": "ITAdmin",
        "department": "IT",
    },
    "key-it-security-1": {
        "user_id": "u_itsec_1",
        "role": "ITSecurity",
        "department": "IT",
    },

    # ============================================================
    # 🟫 EXECUTIVE / LEADERSHIP TEAM
    # ============================================================
    "key-exec-1": {
        "user_id": "u_exec_1",
        "role": "Executive",
        "department": "Executive",
    },
    "key-ceo-1": {
        "user_id": "u_ceo_1",
        "role": "CEO",
        "department": "Executive",
    },
    "key-cto-1": {
        "user_id": "u_cto_1",
        "role": "CTO",
        "department": "Executive",
    },

    # ============================================================
    # ⬜ GUEST / TEMP ACCESS
    # ============================================================
    "key-guest-1": {
        "user_id": "u_guest_1",
        "role": "Guest",
        "department": "General",
    },
}


def get_user_from_api_key(key: str) -> Optional[Dict[str, Any]]:
    """Legacy function for backward compatibility with API keys."""
    return _API_KEYS.get(key)


def create_access_token(user_data: Dict[str, Any]) -> str:
    """
    Create a JWT access token for a user.
    
    Args:
        user_data: Dict containing user_id, username, role, department
        
    Returns:
        JWT token string
    """
    # Calculate expiration time
    expire = datetime.utcnow() + timedelta(days=JWT_EXPIRATION_DAYS)
    
    # Create token payload
    payload = {
        "user_id": user_data["user_id"],
        "username": user_data.get("username"),
        "role": user_data["role"],
        "department": user_data["department"],
        "exp": expire,
        "iat": datetime.utcnow()
    }
    
    # Encode token
    token = jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    
    logger.info(f"Created access token for user: {user_data.get('username')} (expires in {JWT_EXPIRATION_DAYS} days)")
    return token


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload if valid
        None if token is invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        return None
