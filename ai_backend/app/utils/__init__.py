# Legacy utils - DEPRECATED
# Use app.modules.core.utils instead

import warnings
warnings.warn(
    "app.utils is deprecated. Use app.modules.core.utils instead.",
    DeprecationWarning,
    stacklevel=2
)

# Import from legacy location for backward compatibility
from app.services.legacy.utils.doc_parser import *