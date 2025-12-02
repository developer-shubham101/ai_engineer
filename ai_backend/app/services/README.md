# Legacy Services Archive

This directory contains legacy services that have been replaced by the new modular architecture in `app/modules/`.

## Migration Status

### ✅ Migrated to Modules
- `auth.py` → `app/modules/auth/jwt_auth.py` (REMOVED)
- `user_service.py` → `app/modules/auth/user_manager.py` (REMOVED)  
- `support_chat.py` → `app/modules/auth/session_manager.py` (REMOVED)
- `sentiment_classifier.py` → `app/modules/core/utils.py` (REMOVED)
- `utility.py` → `app/modules/core/utils.py` (MOVED TO LEGACY)
- `profile_analyzer.py` → `app/modules/core/profile_analyzer.py` (MOVED TO LEGACY)

### 🔄 Pending Migration
- `base_rag_service.py` → `app/modules/llm/rag_orchestrator.py`
- `*_rag_service.py` → `app/modules/llm/providers.py`
- `model_manager.py` → `app/modules/llm/model_manager.py`
- `version_tracking.py` → `app/modules/core/version_manager.py`
- `chroma_utils.py` → `app/modules/vector_db/chroma_impl.py`

## Usage

**DO NOT** import from legacy files. Use the new modular architecture:

```python
# OLD (deprecated)
from app.services.legacy.auth import create_access_token

# NEW (modular)
from app.modules.integration import get_container

container = get_container()
container.initialize()
authenticator = container.get_authenticator()
token = await authenticator.create_access_token(user_data)
```

## Legacy Directory

Files in `legacy/` are kept for reference during transition and will be removed once migration is complete.