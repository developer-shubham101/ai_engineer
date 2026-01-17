"""Custom LLM implementation for CrewAI with third-party API support."""

import logging
from typing import Dict, List

import requests
from crewai.llm import BaseLLM

logger = logging.getLogger(__name__)


class CustomLLM(BaseLLM):
    """Custom LLM class for CrewAI that works with third-party /ask endpoints."""

    def __init__(self, base_url: str, api_key: str = None, **kwargs):
        super().__init__(model="CustomLLM")
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})

        # Set default parameters
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 512)

    def call(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make API call to custom third-party endpoint."""
        try:
            # Convert messages to single prompt
            prompt = self._messages_to_prompt(messages)

            payload = {
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "top_p": kwargs.get("top_p", 0.9)
            }

            logger.info(f"CustomLLM payload: {payload}")

            response = self.session.post(
                f"{self.base_url}/ask",
                json=payload,
                timeout=300
            )
            response.raise_for_status()

            result = response.json()
            logger.info(f"CustomLLM response: {result}")
            if result.get("success"):
                return result.get("response", "")
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"CustomLLM API error: {error_msg}")
                return f"Error: {error_msg}"

        except Exception as e:
            logger.error(f"CustomLLM call failed: {e}")
            return f"Request failed: {str(e)}"

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to single prompt string."""
        if not messages:
            return "Assistant:"

        prompt_parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        return "\n\n".join(prompt_parts) + "\n\nAssistant:"
