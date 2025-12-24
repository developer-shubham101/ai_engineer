"""Custom LLM implementation for CrewAI with ColabLLM support."""

import logging
from typing import Any, Dict, List, Optional

import requests
from crewai.llm import BaseLLM

logger = logging.getLogger(__name__)


class ColabLLM(BaseLLM):
    """Custom LLM class for CrewAI that works with ColabLLM /ask endpoint."""
    
    def __init__(self, base_url: str, api_key: str = None, **kwargs):
        super().__init__(model="ColabLLM")
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"Authorization": f"Bearer {api_key}"})
        
        # Set default parameters
        self.temperature = kwargs.get('temperature', 0.7)
        self.max_tokens = kwargs.get('max_tokens', 256)
    
    def call(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Make API call to ColabLLM endpoint."""
        try:
            # Convert messages to single prompt
            prompt = self._messages_to_prompt(messages)
            
            payload = {
                "prompt": prompt,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", 0.9),
                "n_ctx": kwargs.get("n_ctx", 4096),
                "n_threads": kwargs.get("n_threads", 4),
                "n_batch": kwargs.get("n_batch", 128)
            }
            
            response = self.session.post(
                f"{self.base_url}/ask",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            
            result = response.json()
            if result.get("success"):
                return result.get("response", "")
            else:
                error_msg = result.get("error", "Unknown error")
                logger.error(f"ColabLLM API error: {error_msg}")
                return f"Error: {error_msg}"
                
        except Exception as e:
            logger.error(f"ColabLLM call failed: {e}")
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