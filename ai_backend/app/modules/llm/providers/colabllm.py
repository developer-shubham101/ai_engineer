"""ColabLLM provider implementation."""

import logging
from typing import Dict, Any

import requests

from ..interfaces import ILLMProvider, LLMResponse

logger = logging.getLogger(__name__)


class ColabLLMProvider(ILLMProvider):
    """ColabLLM provider using /ask endpoint."""

    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

    async def generate(
            self,
            prompt: str,
            max_tokens: int = 128,
            temperature: float = 0.6,
            **kwargs
    ) -> LLMResponse:
        """Generate response using ColabLLM /ask endpoint."""
        try:
            payload = {
                "prompt": prompt,
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
                logger.error(f"ColabLLM API error: {error_msg}")
                text = f"Error: {error_msg}"
            
            return LLMResponse(
                text=text,
                metadata={"provider": "colabllm"}
            )

        except requests.exceptions.RequestException as e:
            logger.error(f"ColabLLM request failed: {e}")
            return LLMResponse(
                text=f"Request failed: {str(e)}",
                metadata={"provider": "colabllm", "error": str(e)}
            )
        except Exception as e:
            logger.error(f"ColabLLM generation failed: {e}")
            return LLMResponse(
                text=f"Generation failed: {str(e)}",
                metadata={"provider": "colabllm", "error": str(e)}
            )

    def get_provider_name(self) -> str:
        """Get provider name."""
        return "colabllm"

    def get_model_name(self) -> str:
        """Get current model name."""
        return "colabllm-model"

    def is_available(self) -> bool:
        """Check if provider is available."""
        return True

    def get_max_context_length(self) -> int:
        """Get maximum context length."""
        return 2048

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        return {
            "provider": "colabllm",
            "base_url": self.base_url,
            "supports_streaming": False,
            "max_context": 2048
        }
