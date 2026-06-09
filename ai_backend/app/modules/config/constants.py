"""Application constants and enums."""

from enum import Enum
from typing import Dict, List, Set
from .settings import settings


class UserRole(Enum):
    """User role enumeration."""
    SUPER_ADMIN = "SuperAdmin"
    MANAGER = "Manager"
    HR = "HR"
    EMPLOYEE = "Employee"
    PUBLIC_USER = "PublicUser"
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
    """LLM provider enumeration.

    Supported providers:
    - GOOGLE: Google Gemini API models
    - GPT/OPENAI: OpenAI GPT models via API
    - HUGGINGFACE/HF: Hugging Face Inference API models
    - COLABLLM: Custom API endpoints (legacy name for backward compatibility)
    - CUSTOMLLM: Custom/third-party API endpoints (preferred)
    - LLAMASERVER: llama-server.exe with OpenAI-compatible API

    NOTE: LOCAL provider archived — see archive/local_llm/
    """
    # LOCAL = "local"  # ARCHIVED — see archive/local_llm/
    GOOGLE = "google"
    GPT = "gpt"
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"
    HF = "hf"
    COLABLLM = "colabllm"  # Legacy name for backward compatibility
    CUSTOMLLM = "customllm"  # Custom/third-party LLM APIs
    LLAMASERVER = "llamaserver"


class DocumentStatus(Enum):
    """Document status enumeration."""
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


# Role hierarchy levels
ROLE_LEVELS: Dict[str, int] = {
    UserRole.SUPER_ADMIN.value: 4,
    UserRole.MANAGER.value: 3,
    UserRole.HR.value: 2,
    UserRole.EMPLOYEE.value: 1,
    UserRole.PUBLIC_USER.value: 0,
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

# Valid sets for validation
VALID_DEPARTMENTS: Set[str] = {dept.value for dept in Department}
VALID_ROLES: Set[str] = {role.value for role in UserRole}
VALID_SENSITIVITY_LEVELS: Set[str] = {level.value for level in SensitivityLevel}
VALID_DOCUMENT_STATUSES: Set[str] = {status.value for status in DocumentStatus}
VALID_PROVIDERS: Set[str] = {provider.value for provider in LLMProvider}

# Default values
DEFAULT_TOP_K = 3
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.1
DEFAULT_EMBEDDING_MODEL = settings.EMBEDDING_MODEL_KEY
DEFAULT_DEPARTMENT = Department.GENERAL.value
DEFAULT_SENSITIVITY = SensitivityLevel.PUBLIC_INTERNAL.value
DEFAULT_STATUS = DocumentStatus.PUBLISHED.value

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
MAX_HISTORY_TURNS = 5

# File handling
MAX_FILE_SIZE_MB = 5
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
SUPPORTED_EXTENSIONS = [".txt", ".md", ".markdown", ".html", ".htm", ".json", ".csv"]
MARKDOWN_EXTENSIONS = [".md", ".markdown"]
HTML_EXTENSIONS = [".html", ".htm"]
JSON_EXTENSIONS = [".json"]

# HTTP status messages
HTTP_MESSAGES = {
    "UNAUTHORIZED": "Could not validate credentials",
    "FORBIDDEN": "Access denied",
    "FILE_EMPTY": "Uploaded file is empty",
    "FILE_TOO_LARGE": f"File too large (max {MAX_FILE_SIZE_MB} MB)",
    "FILE_DECODE_ERROR": "Failed to decode file; ensure it's a text file (UTF-8)",
    "COLLECTION_CLEARED": "Collection cleared"
}

# RBAC constants
HR_LEVEL_THRESHOLD = 2
SUPER_ADMIN_ROLES = [UserRole.SUPER_ADMIN.value]
MANAGER_PLUS_ROLES = [UserRole.SUPER_ADMIN.value, UserRole.MANAGER.value]
HR_PLUS_ROLES = [UserRole.SUPER_ADMIN.value, UserRole.MANAGER.value, UserRole.HR.value]
EMPLOYEE_PLUS_ROLES = [UserRole.SUPER_ADMIN.value, UserRole.MANAGER.value, UserRole.HR.value, UserRole.EMPLOYEE.value]