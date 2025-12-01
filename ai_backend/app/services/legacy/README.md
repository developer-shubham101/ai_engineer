# Legacy Services Archive

This directory contains legacy services that have been replaced by the new modular architecture.

## Replaced Services

- `auth.py` → `app/modules/auth/jwt_auth.py`
- `user_service.py` → `app/modules/auth/user_manager.py`
- `support_chat.py` → `app/modules/auth/session_manager.py`
- `profile_analyzer.py` → `app/modules/core/profile_analyzer.py`
- `sentiment_classifier.py` → Integrated into session manager
- `utility.py` → `app/modules/core/utils.py`
- `chroma_utils.py` → `app/modules/vector_db/chroma_impl.py`

## Migration Status

These files are kept for reference during the transition period. They will be removed once all functionality has been fully migrated to the modular architecture.

## Usage

Do not import from these legacy files. Use the new modular architecture instead:

```python
# OLD (deprecated)
from app.services.auth import create_access_token

# NEW (modular)
from app.modules.integration import get_container
authenticator = get_container().get_authenticator()
token = await authenticator.create_token(user_data)
```