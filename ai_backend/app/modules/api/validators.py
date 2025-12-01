"""API input validators."""

from typing import Dict, Any, List, Optional
import re


class ValidationError(Exception):
    """Custom validation error."""
    pass


class APIValidator:
    """Base API validator."""
    
    @staticmethod
    def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> None:
        """Validate required fields are present."""
        missing = [field for field in required_fields if field not in data or data[field] is None]
        if missing:
            raise ValidationError(f"Missing required fields: {', '.join(missing)}")
    
    @staticmethod
    def validate_string_length(value: str, field_name: str, min_len: int = 1, max_len: int = 10000) -> None:
        """Validate string length."""
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string")
        if len(value) < min_len:
            raise ValidationError(f"{field_name} must be at least {min_len} characters")
        if len(value) > max_len:
            raise ValidationError(f"{field_name} must be at most {max_len} characters")
    
    @staticmethod
    def validate_numeric_range(value: float, field_name: str, min_val: float, max_val: float) -> None:
        """Validate numeric value is within range."""
        if not isinstance(value, (int, float)):
            raise ValidationError(f"{field_name} must be a number")
        if value < min_val or value > max_val:
            raise ValidationError(f"{field_name} must be between {min_val} and {max_val}")


class DocumentValidator(APIValidator):
    """Validator for document-related requests."""
    
    VALID_SENSITIVITY_LEVELS = [
        "public_internal", "department_confidential", "role_confidential", 
        "highly_confidential", "super_confidential", "personal"
    ]
    
    VALID_DEPARTMENTS = [
        "HR", "Finance", "Engineering", "IT", "Legal", "Executive", "Admin", "General"
    ]
    
    @classmethod
    def validate_document_metadata(cls, metadata: Dict[str, Any]) -> None:
        """Validate document metadata."""
        # Validate sensitivity level
        sensitivity = metadata.get("sensitivity")
        if sensitivity and sensitivity not in cls.VALID_SENSITIVITY_LEVELS:
            raise ValidationError(f"Invalid sensitivity level: {sensitivity}")
        
        # Validate department
        department = metadata.get("department")
        if department and department not in cls.VALID_DEPARTMENTS:
            raise ValidationError(f"Invalid department: {department}")
        
        # Validate allowed_roles if present
        allowed_roles = metadata.get("allowed_roles")
        if allowed_roles and not isinstance(allowed_roles, list):
            raise ValidationError("allowed_roles must be a list")


class UserValidator(APIValidator):
    """Validator for user-related requests."""
    
    VALID_ROLES = ["SuperAdmin", "Manager", "HR", "Employee", "Guest"]
    
    @staticmethod
    def validate_username(username: str) -> None:
        """Validate username format."""
        if not re.match(r'^[a-zA-Z0-9_]{3,50}$', username):
            raise ValidationError("Username must be 3-50 characters, alphanumeric and underscore only")
    
    @classmethod
    def validate_role(cls, role: str) -> None:
        """Validate user role."""
        if role not in cls.VALID_ROLES:
            raise ValidationError(f"Invalid role: {role}. Must be one of: {', '.join(cls.VALID_ROLES)}")


class QueryValidator(APIValidator):
    """Validator for query requests."""
    
    @classmethod
    def validate_query_request(cls, data: Dict[str, Any]) -> None:
        """Validate query request data."""
        cls.validate_required_fields(data, ["question"])
        cls.validate_string_length(data["question"], "question", min_len=1, max_len=1000)
        
        if "top_k" in data:
            cls.validate_numeric_range(data["top_k"], "top_k", 1, 20)
        
        if "max_tokens" in data:
            cls.validate_numeric_range(data["max_tokens"], "max_tokens", 1, 4096)
        
        if "temperature" in data:
            cls.validate_numeric_range(data["temperature"], "temperature", 0.0, 1.0)