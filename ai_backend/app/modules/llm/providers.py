"""LLM provider implementations."""

from typing import Dict, Any, Optional
import logging

from .interfaces import ILLMProvider, LLMResponse
from ..config.settings import settings

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


class LocalLLMProvider(ILLMProvider):
    """Local LLM provider (placeholder for actual implementation)."""
    
    def __init__(self, model_name: str = "mistral-7b-instruct-v0.2"):
        self.model_name = model_name
        self.provider_name = "local"
        self.model = None  # Would load actual model here
    
    async def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.1, **kwargs) -> LLMResponse:
        """Generate response using local model."""
        # Placeholder - would use actual local model
        response_text = f"Local model response (placeholder)"
        
        return LLMResponse(
            text=response_text,
            metadata={
                "model": self.model_name,
                "provider": self.provider_name,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
    
    def get_provider_name(self) -> str:
        return self.provider_name
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def is_available(self) -> bool:
        # Check if model file exists
        model_path = settings.MODELS_DIR / f"{self.model_name}.gguf"
        return model_path.exists()
    
    def get_max_context_length(self) -> int:
        return 4096


class OpenAIProvider(ILLMProvider):
    """OpenAI LLM provider (placeholder)."""
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.provider_name = "openai"
        self.api_key = settings.OPENAI_API_KEY
    
    async def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.1, **kwargs) -> LLMResponse:
        """Generate response using OpenAI API."""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        # Placeholder - would use actual OpenAI API
        response_text = f"OpenAI response (placeholder)"
        
        return LLMResponse(
            text=response_text,
            metadata={
                "model": self.model_name,
                "provider": self.provider_name,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
        )
    
    def get_provider_name(self) -> str:
        return self.provider_name
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def get_max_context_length(self) -> int:
        return 4096


class ProviderFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create_provider(provider_name: str, model_name: Optional[str] = None) -> ILLMProvider:
        """Create LLM provider by name."""
        if provider_name == "mock":
            return MockLLMProvider(model_name or "mock-model")
        elif provider_name == "local":
            return LocalLLMProvider(model_name or settings.DEFAULT_MODEL_NAME)
        elif provider_name == "openai":
            return OpenAIProvider(model_name or "gpt-3.5-turbo")
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    @staticmethod
    def get_available_providers() -> Dict[str, bool]:
        """Get available providers and their status."""
        providers = {}
        
        for provider_name in ["mock", "local", "openai"]:
            try:
                provider = ProviderFactory.create_provider(provider_name)
                providers[provider_name] = provider.is_available()
            except Exception:
                providers[provider_name] = False
        
        return providers