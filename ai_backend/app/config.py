# Legacy config file - DEPRECATED
# Use app.modules.config.settings instead

import warnings
warnings.warn(
    "app.config is deprecated. Use app.modules.config.settings instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from new modular config for backward compatibility
from app.modules.config.settings import settings
from app.modules.config.constants import *