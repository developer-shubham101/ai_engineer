"""LlamaServer provider plugin using OpenAI-compatible API."""

import logging
from typing import Dict, Any, Optional

from .provider_factory import ProviderPlugin

logger = logging.getLogger(__name__)


class LlamaServerProviderPlugin(ProviderPlugin):
    """LlamaServer provider plugin using OpenAI-compatible API."""
    
    @property
    def name(self) -> str:
        return "llamaserver"
    
    @property
    def requires_api_key(self) -> bool:
        return False
    
    async def create_provider(self, config: Optional[Dict[str, Any]] = None):
        from .providers.llamaserver import LlamaServerProvider
        return LlamaServerProvider(config or {})
    
    def is_available(self) -> bool:
        try:
            from autogen_ext.models.openai import OpenAIChatCompletionClient
            return True
        except ImportError:
            return False