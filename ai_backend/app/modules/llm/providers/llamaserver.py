"""LlamaServer provider implementation using OpenAI-compatible API."""

import logging
from typing import Dict, Any

from autogen_ext.models.openai import OpenAIChatCompletionClient

from ..interfaces import ILLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class LlamaServerProvider(ILLMProvider):
    """LlamaServer provider using OpenAI-compatible API."""

    def __init__(self, configs: Dict[str, Any]):
        from ...config.settings import settings
        self.base_url = configs.get("base_url", settings.LLAMASERVER_BASE_URL)
        self.model_name = configs.get("model_name", settings.LLAMASERVER_MODEL_NAME)
        
        self.client = OpenAIChatCompletionClient(
            model=self.model_name,
            base_url=self.base_url,
            api_key="placeholder",
            model_info={
                "vision": False,
                "function_calling": False,
                "json_output": False,
                "structured_output": False,
                "family": "unknown",
            },
        )

    async def generate(
            self,
            prompt: str,
            max_tokens: int = 256,
            temperature: float = 0.7,
            **kwargs
    ) -> LLMResponse:
        """Generate response using llama-server."""
        try:
            from autogen_agentchat.messages import UserMessage
            messages = [UserMessage(content=prompt, source="user")]
            
            response = await self.client.create(
                messages=messages
            )
            
            text = response.content if hasattr(response, 'content') else str(response)
            
            return LLMResponse(
                text=text,
                metadata={"provider": "llamaserver", "model": self.model_name}
            )

        except Exception as e:
            logger.error(f"LlamaServer generation failed: {e}")
            return LLMResponse(
                text=f"Generation failed: {str(e)}",
                metadata={"provider": "llamaserver", "error": str(e)}
            )

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "llamaserver"

    def get_model_name(self) -> str:
        """Get current model name."""
        return self.model_name

    def is_available(self) -> bool:
        """Check if provider is available."""
        return True

    def get_max_context_length(self) -> int:
        """Get maximum context length."""
        return 4096

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": "llamaserver",
            "model": self.model_name,
            "base_url": self.base_url,
            "supports_streaming": False,
            "max_context": 4096
        }