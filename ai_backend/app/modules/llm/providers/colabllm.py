"""Custom third-party LLM provider implementation."""

import logging
from typing import Dict, Any, Union, List

import requests

from ..interfaces import ILLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class CustomLLMProvider(ILLMProvider):
    """Custom third-party LLM provider using /ask endpoint."""

    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    async def generate(
            self,
            prompt: Union[str, List[Dict[str, str]]],
            max_tokens: int = 128,
            temperature: float = 0.6,
            **kwargs
    ) -> LLMResponse:
        """Generate response using custom third-party /ask endpoint.
        Accepts either a string prompt or a list of message dicts with 'role' and 'content'.
        """
        try:
            # Convert messages to prompt if needed
            if isinstance(prompt, list):
                logger.debug("Converting messages to prompt for CustomLLM")
                prompt_str = self._messages_to_prompt(prompt)
            else:
                prompt_str = prompt
            
            payload = {
                "prompt": prompt_str,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": kwargs.get("top_p", 0.9),
                "n_ctx": kwargs.get("n_ctx", 2048),
                "n_threads": kwargs.get("n_threads", 4),
                "n_batch": kwargs.get("n_batch", 128)
            }

            response = self.session.post(
                f"{self.base_url}/ask",
                json=payload,
                timeout=3000
            )
            response.raise_for_status()

            result = response.json()
            if result.get("success"):
                text = result.get("response", "")
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"CustomLLM API error: {error_msg}")
                text = f"Error: {error_msg}"
            
            return LLMResponse(
                text=text,
                metadata={"provider": "customllm"}
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"CustomLLM request failed: {e}")
            return LLMResponse(
                text=f"Request failed: {str(e)}",
                metadata={"provider": "customllm", "error": str(e)}
            )
        except Exception as e:
            logger.error(f"CustomLLM generation failed: {e}")
            return LLMResponse(
                text=f"Generation failed: {str(e)}",
                metadata={"provider": "customllm", "error": str(e)}
            )

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "customllm"

    def get_model_name(self) -> str:
        """Get current model name."""
        return "customllm-model"

    def is_available(self) -> bool:
        """Check if provider is available."""
        return True

    def get_max_context_length(self) -> int:
        """Get maximum context length."""
        return 2048

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": "customllm",
            "base_url": self.base_url,
            "supports_streaming": False,
            "max_context": 2048
        }
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages array to single prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        return "\n\n".join(prompt_parts)


# Backward compatibility alias
ColabLLMProvider = CustomLLMProvider
