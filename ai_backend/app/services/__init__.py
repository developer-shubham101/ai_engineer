# Services package
# 
# This package contains both active services and legacy services.
# New development should use the modular architecture in app/modules/

# Active services (still in use)
# Active services (still in use)
from . import base_rag_service
# from . import rag_local_service # Moved to app/modules/llm/providers/local.py
# from . import google_models # Moved to app/modules/llm/providers/google.py
# from . import gpt_rag_service # Moved to app/modules/llm/providers/openai.py
# from . import hf_rag_service # Moved to app/modules/llm/providers/huggingface.py
from . import model_manager
from . import model_manager
from . import local_model_manager
from . import prompt_builder
from .legacy import version_tracking, chroma_utils

# Legacy services moved to legacy/ directory
# These are deprecated and will be removed in future versions
# Use app/modules/ instead for new development