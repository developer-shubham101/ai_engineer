"""Configuration module exports."""

from .constants import (
    # Enums
    UserRole, SensitivityLevel, Department, LLMProvider, DocumentStatus,
    
    # Validation sets
    VALID_DEPARTMENTS, VALID_ROLES, VALID_SENSITIVITY_LEVELS, VALID_DOCUMENT_STATUSES, VALID_PROVIDERS,
    
    # Hierarchy mappings
    ROLE_LEVELS, SENSITIVITY_LEVELS,
    
    # Defaults
    DEFAULT_TOP_K, DEFAULT_MAX_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_EMBEDDING_MODEL,
    DEFAULT_DEPARTMENT, DEFAULT_SENSITIVITY, DEFAULT_STATUS,
    
    # API prefixes
    API_PREFIX, RAG_PREFIX, AUTH_PREFIX, MODELS_PREFIX, TRAINING_PREFIX,
    
    # File handling
    MAX_FILE_SIZE_MB, MAX_FILE_SIZE_BYTES, SUPPORTED_EXTENSIONS,
    MARKDOWN_EXTENSIONS, HTML_EXTENSIONS, JSON_EXTENSIONS,
    
    # HTTP messages
    HTTP_MESSAGES,
    
    # RBAC constants
    HR_LEVEL_THRESHOLD, SUPER_ADMIN_ROLES, MANAGER_PLUS_ROLES, HR_PLUS_ROLES, EMPLOYEE_PLUS_ROLES,
    
    # Token limits
    MAX_PROMPT_TOKENS, MAX_CONTEXT_TOKENS, MAX_SYSTEM_TOKENS, MAX_HISTORY_TURNS
)

from .models import (
    ModelConfig, ProviderConfig, DatabaseConfig, SecurityConfig, 
    EmbeddingConfig, RAGConfig, ValidationConfig, AppConfig
)

from .settings import settings

__all__ = [
    # Enums
    "UserRole", "SensitivityLevel", "Department", "LLMProvider", "DocumentStatus",
    
    # Validation
    "VALID_DEPARTMENTS", "VALID_ROLES", "VALID_SENSITIVITY_LEVELS", "VALID_DOCUMENT_STATUSES", "VALID_PROVIDERS",
    
    # Hierarchies
    "ROLE_LEVELS", "SENSITIVITY_LEVELS",
    
    # Defaults
    "DEFAULT_TOP_K", "DEFAULT_MAX_TOKENS", "DEFAULT_TEMPERATURE", "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_DEPARTMENT", "DEFAULT_SENSITIVITY", "DEFAULT_STATUS",
    
    # API
    "API_PREFIX", "RAG_PREFIX", "AUTH_PREFIX", "MODELS_PREFIX", "TRAINING_PREFIX",
    
    # Files
    "MAX_FILE_SIZE_MB", "MAX_FILE_SIZE_BYTES", "SUPPORTED_EXTENSIONS",
    "MARKDOWN_EXTENSIONS", "HTML_EXTENSIONS", "JSON_EXTENSIONS",
    
    # Messages
    "HTTP_MESSAGES",
    
    # RBAC
    "HR_LEVEL_THRESHOLD", "SUPER_ADMIN_ROLES", "MANAGER_PLUS_ROLES", "HR_PLUS_ROLES", "EMPLOYEE_PLUS_ROLES",
    
    # Tokens
    "MAX_PROMPT_TOKENS", "MAX_CONTEXT_TOKENS", "MAX_SYSTEM_TOKENS", "MAX_HISTORY_TURNS",
    
    # Models
    "ModelConfig", "ProviderConfig", "DatabaseConfig", "SecurityConfig", 
    "EmbeddingConfig", "RAGConfig", "ValidationConfig", "AppConfig",
    
    # Settings
    "settings"
]