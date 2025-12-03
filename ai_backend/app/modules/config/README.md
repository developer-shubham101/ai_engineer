# Configuration Module

This module centralizes all configuration constants, settings, and models for the AI Backend system.

## Structure

- `constants.py` - Application constants, enums, and validation sets
- `models.py` - Pydantic configuration models
- `settings.py` - Environment-based settings
- `database_config.py` - Database configuration
- `local_models.json` - Local model definitions
- `onboarding_fields.json` - User onboarding field definitions

## Usage

### Import Constants

```python
from app.modules.config import (
    UserRole, Department, SensitivityLevel,
    VALID_ROLES, VALID_DEPARTMENTS, VALID_SENSITIVITY_LEVELS,
    DEFAULT_TOP_K, DEFAULT_TEMPERATURE,
    HTTP_MESSAGES, EMPLOYEE_PLUS_ROLES
)

# Use enums for type safety
user_role = UserRole.EMPLOYEE.value

# Use validation sets
if department in VALID_DEPARTMENTS:
    # Valid department
    pass

# Use default values
top_k = DEFAULT_TOP_K

# Use HTTP messages
raise HTTPException(status_code=400, detail=HTTP_MESSAGES["FILE_EMPTY"])

# Use role groups
@router.post("/endpoint", dependencies=[Depends(require_roles(EMPLOYEE_PLUS_ROLES))])
```

### Import Settings

```python
from app.modules.config import settings

# Access paths
models_dir = settings.MODELS_DIR
database_dir = settings.DATABASE_DIR

# Access configuration
jwt_secret = settings.JWT_SECRET_KEY
embedding_model = settings.EMBEDDING_MODEL_NAME
```

### Import Configuration Models

```python
from app.modules.config import AppConfig, RAGConfig, SecurityConfig

# Create configuration instances
rag_config = RAGConfig(default_top_k=5, max_context_length=4096)
security_config = SecurityConfig(jwt_expiration_days=30)
```

## Benefits

1. **Centralized Configuration** - All constants in one place
2. **Type Safety** - Enums prevent typos and invalid values
3. **Easy Maintenance** - Change values in one location
4. **Validation** - Pre-defined sets for input validation
5. **Environment Support** - Settings from environment variables
6. **Documentation** - Clear structure and usage examples

## Migration from Hardcoded Strings

Before:
```python
# Hardcoded strings scattered throughout codebase
ALLOWED_ROLES = {"SuperAdmin", "Manager", "HR", "Employee"}
if user_role not in ALLOWED_ROLES:
    raise HTTPException(status_code=403, detail="Access denied")

@router.post("/endpoint", dependencies=[Depends(require_roles(["SuperAdmin", "Manager", "HR", "Employee"]))])
```

After:
```python
# Centralized constants
from app.modules.config import VALID_ROLES, EMPLOYEE_PLUS_ROLES, HTTP_MESSAGES

if user_role not in VALID_ROLES:
    raise HTTPException(status_code=403, detail=HTTP_MESSAGES["FORBIDDEN"])

@router.post("/endpoint", dependencies=[Depends(require_roles(EMPLOYEE_PLUS_ROLES))])
```

## Adding New Constants

1. **Add to appropriate enum** in `constants.py`
2. **Update validation sets** if needed
3. **Add to `__init__.py`** exports
4. **Update this README** with usage examples

Example:
```python
# In constants.py
class NewCategory(Enum):
    CATEGORY_A = "category_a"
    CATEGORY_B = "category_b"

VALID_CATEGORIES: Set[str] = {cat.value for cat in NewCategory}

# In __init__.py
from .constants import NewCategory, VALID_CATEGORIES

__all__ = [
    # ... existing exports
    "NewCategory", "VALID_CATEGORIES"
]
```