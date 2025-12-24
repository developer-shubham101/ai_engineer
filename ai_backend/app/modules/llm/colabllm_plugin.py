"""ColabLLM provider plugin."""

import os
from typing import Dict, Any, Optional
from .provider_factory import ProviderPlugin


class ColabLLMProviderPlugin(ProviderPlugin):
    """ColabLLM provider plugin."""
    
    @property
    def name(self) -> str:
        return "colabllm"
    
    @property
    def requires_api_key(self) -> bool:
        return False  # API key is optional
    
    async def create_provider(self, config: Optional[Dict[str, Any]] = None):
        from app.modules.llm.providers.colabllm import ColabLLMProvider
        from app.modules.config.settings import settings
        
        base_url = config.get("base_url") if config else None
        api_key = config.get("api_key") if config else None
        
        if not base_url:
            base_url = settings.COLABLLM_BASE_URL
        if not api_key:
            api_key = settings.COLABLLM_API_KEY
            
        return ColabLLMProvider(base_url=base_url, api_key=api_key)
    
    def is_available(self) -> bool:
        # Always available since it's a local service
        return True