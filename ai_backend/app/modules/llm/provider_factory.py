"""Improved provider factory with plugin architecture."""

from typing import Dict, Type, Optional, Any
from abc import ABC, abstractmethod
import logging

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()

class ProviderPlugin(ABC):
    """Base class for provider plugins."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass
    
    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """Whether provider requires API key."""
        pass
    
    @abstractmethod
    async def create_provider(self, config: Optional[Dict[str, Any]] = None):
        """Create provider instance."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass


class LocalProviderPlugin(ProviderPlugin):
    """Local LLM provider plugin."""
    
    @property
    def name(self) -> str:
        return "local"
    
    @property
    def requires_api_key(self) -> bool:
        return False
    
    async def create_provider(self, config: Optional[Dict[str, Any]] = None):
        from app.modules.llm.providers.local import LocalLLMProvider
        model_name = config.get("model_name") if config else None
        return LocalLLMProvider(model_name=model_name)
    
    def is_available(self) -> bool:
        try:
            from app.modules.llm.providers.local import LocalLLMProvider
            return True
        except ImportError:
            return False


class GoogleProviderPlugin(ProviderPlugin):
    """Google provider plugin."""
    
    @property
    def name(self) -> str:
        return "google"
    
    @property
    def requires_api_key(self) -> bool:
        return True
    
    async def create_provider(self, config: Optional[Dict[str, Any]] = None):
        from app.modules.llm.providers.google import GoogleLLMProvider
        return GoogleLLMProvider()
    
    def is_available(self) -> bool:
        import os
        google_api_key = os.getenv("GOOGLE_API_KEY")
        return bool(google_api_key)


class ProviderRegistry:
    """Registry for provider plugins."""
    
    def __init__(self):
        self._plugins: Dict[str, ProviderPlugin] = {}
        self._register_default_plugins()
    
    def _register_default_plugins(self):
        """Register default provider plugins."""
        plugins = [
            LocalProviderPlugin(),
            GoogleProviderPlugin(),
        ]
        
        # Register ColabLLM plugin
        try:
            from .colabllm_plugin import ColabLLMProviderPlugin
            plugins.append(ColabLLMProviderPlugin())
        except ImportError:
            logger.warning("ColabLLM plugin not available")
        
        # Register LlamaServer plugin
        try:
            from .llamaserver_plugin import LlamaServerProviderPlugin
            plugins.append(LlamaServerProviderPlugin())
        except ImportError:
            logger.warning("LlamaServer plugin not available")
        
        for plugin in plugins:
            self.register(plugin)
    
    def register(self, plugin: ProviderPlugin):
        """Register a provider plugin."""
        self._plugins[plugin.name] = plugin
        logger.info(f"Registered provider plugin: {plugin.name}")
    
    def unregister(self, name: str):
        """Unregister a provider plugin."""
        if name in self._plugins:
            del self._plugins[name]
            logger.info(f"Unregistered provider plugin: {name}")
    
    def get_plugin(self, name: str) -> Optional[ProviderPlugin]:
        """Get provider plugin by name."""
        return self._plugins.get(name)
    
    def list_available(self) -> Dict[str, bool]:
        """List all plugins and their availability."""
        return {name: plugin.is_available() for name, plugin in self._plugins.items()}
    
    async def create_provider(self, name: str, config: Optional[Dict[str, Any]] = None):
        """Create provider instance."""
        plugin = self.get_plugin(name)
        if not plugin:
            raise ValueError(f"Unknown provider: {name}")
        
        if not plugin.is_available():
            raise RuntimeError(f"Provider {name} is not available")
        
        return await plugin.create_provider(config)


# Global registry instance
_registry = ProviderRegistry()


def get_provider_registry() -> ProviderRegistry:
    """Get global provider registry."""
    return _registry


async def create_provider(name: str, config: Optional[Dict[str, Any]] = None):
    """Convenience function to create provider."""
    return await _registry.create_provider(name, config)