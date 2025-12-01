"""Application constants and enums."""

from enum import Enum
from typing import Dict, List


class UserRole(Enum):
    """User role enumeration."""
    SUPER_ADMIN = "SuperAdmin"
    MANAGER = "Manager"
    HR = "HR"
    EMPLOYEE = "Employee"
    GUEST = "Guest"


class SensitivityLevel(Enum):
    """Document sensitivity level enumeration."""
    SUPER_CONFIDENTIAL = "super_confidential"
    HIGHLY_CONFIDENTIAL = "highly_confidential"
    ROLE_CONFIDENTIAL = "role_confidential"
    DEPARTMENT_CONFIDENTIAL = "department_confidential"
    PUBLIC_INTERNAL = "public_internal"
    PERSONAL = "personal"


class Department(Enum):
    """Department enumeration."""
    HR = "HR"
    FINANCE = "Finance"
    ENGINEERING = "Engineering"
    IT = "IT"
    LEGAL = "Legal"
    EXECUTIVE = "Executive"
    ADMIN = "Admin"
    GENERAL = "General"


class LLMProvider(Enum):
    """LLM provider enumeration."""
    LOCAL = "local"
    OPENAI = "openai"
    GOOGLE = "google"
    HUGGINGFACE = "huggingface"


# Role hierarchy levels
ROLE_LEVELS: Dict[str, int] = {
    UserRole.SUPER_ADMIN.value: 4,
    UserRole.MANAGER.value: 3,
    UserRole.HR.value: 2,
    UserRole.EMPLOYEE.value: 1,
    UserRole.GUEST.value: 0
}

# Sensitivity levels
SENSITIVITY_LEVELS: Dict[str, int] = {
    SensitivityLevel.SUPER_CONFIDENTIAL.value: 4,
    SensitivityLevel.HIGHLY_CONFIDENTIAL.value: 3,
    SensitivityLevel.ROLE_CONFIDENTIAL.value: 2,
    SensitivityLevel.DEPARTMENT_CONFIDENTIAL.value: 1,
    SensitivityLevel.PUBLIC_INTERNAL.value: 0,
    SensitivityLevel.PERSONAL.value: 1
}

# Valid departments list
VALID_DEPARTMENTS: List[str] = [dept.value for dept in Department]

# Valid roles list
VALID_ROLES: List[str] = [role.value for role in UserRole]

# Valid sensitivity levels list
VALID_SENSITIVITY_LEVELS: List[str] = [level.value for level in SensitivityLevel]

# Default values
DEFAULT_TOP_K = 3
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.1
DEFAULT_EMBEDDING_MODEL = "bge-small-en-v1.5"

# API endpoints
API_PREFIX = "/api"
RAG_PREFIX = f"{API_PREFIX}/rag"
AUTH_PREFIX = f"{API_PREFIX}/auth"
MODELS_PREFIX = f"{API_PREFIX}/models"
TRAINING_PREFIX = f"{API_PREFIX}/training"

# Token limits
MAX_PROMPT_TOKENS = 4096
MAX_CONTEXT_TOKENS = 2048
MAX_SYSTEM_TOKENS = 80

# File extensions for document parsing
SUPPORTED_EXTENSIONS = [
    ".txt", ".md", ".pdf", ".docx", ".json", ".csv"
]