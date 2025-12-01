# Services package
# 
# This package contains both active services and legacy services.
# New development should use the modular architecture in app/modules/

# Active services (still in use)
from . import base_rag_service
from . import rag_local_service
from . import google_models
from . import gpt_rag_service
from . import hf_rag_service
from . import model_manager
from . import local_model_manager
from . import model_training_service
from . import prompt_builder
from . import version_tracking
from . import chroma_utils

# Legacy services moved to legacy/ directory
# These are deprecated and will be removed in future versions
# Use app/modules/ instead for new development