"""LLM provider implementations."""

from typing import Dict, Optional
import logging

from app.modules.llm.providers.google import GoogleLLMProvider
from app.modules.llm.providers.huggingface import HuggingFaceLLMProvider
from app.modules.llm.providers.openai import OpenAILLMProvider
from app.modules.llm.interfaces import ILLMProvider, LLMResponse

# Import the new provider classes

from app.modules.llm.providers.local import LocalLLMProvider

logger = logging.getLogger(__name__)


class MockLLMProvider(ILLMProvider):
    """Mock LLM provider for testing."""
    
    def __init__(self, model_name: str = "mock-model"):
        self.model_name = model_name
        self.provider_name = "mock"
    
    async def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.1, **kwargs) -> LLMResponse:
        """Generate mock response."""
        # Simple mock response based on prompt
        response_text = f"Mock response to: {prompt[:50]}..."
        
        return LLMResponse(
            text=response_text,
            metadata={
                "model": self.model_name,
                "provider": self.provider_name,
                "temperature": temperature,
                "max_tokens": max_tokens
            },
            usage={
                "prompt_tokens": len(prompt) // 4,
                "completion_tokens": len(response_text) // 4,
                "total_tokens": (len(prompt) + len(response_text)) // 4
            },
            finish_reason="completed"
        )
    
    def get_provider_name(self) -> str:
        """Get provider name."""
        return self.provider_name
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name
    
    def is_available(self) -> bool:
        """Check if provider is available."""
        return True
    
    def get_max_context_length(self) -> int:
        """Get maximum context length."""
        return 4096


class ProviderFactory:
    """Factory for creating LLM providers."""
    
    _provider_map = {
        "google": GoogleLLMProvider,
        "openai": OpenAILLMProvider,
        "huggingface": HuggingFaceLLMProvider,
        "local": LocalLLMProvider,
        "mock": MockLLMProvider,
    }

    @staticmethod
    def create_provider(provider_name: str, model_name: Optional[str] = None) -> ILLMProvider:
        """Create LLM provider by name."""
        provider_name = provider_name.lower()
        if provider_name not in ProviderFactory._provider_map:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        provider_class = ProviderFactory._provider_map[provider_name]
        
        if model_name:
            return provider_class(model_name=model_name)
        else:
            return provider_class()

    @staticmethod
    def get_available_providers() -> Dict[str, bool]:
        """Get available providers and their status."""
        providers = {}
        
        for provider_name, provider_class in ProviderFactory._provider_map.items():
            try:
                provider = provider_class()
                providers[provider_name] = provider.is_available()
            except Exception:
                providers[provider_name] = False
        
        return providers
