# app/modules/llm/providers/openai.py
"""
OpenAI LLM provider implementation.
"""

import logging
import os
import time
from typing import Union, List, Dict, Any

from app.modules.llm.interfaces import ILLMProvider, LLMResponse
from app.modules.llm.prompt_builder import estimate_tokens_from_text

try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)

# Optional OpenAI import
try:
    openai.api_key = os.getenv("OPENAI_API_KEY")
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install openai package to use GPT RAG service.")


class OpenAILLMProvider(ILLMProvider):
    """
    OpenAI LLM provider implementation.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        self.model_name = model_name
        self.provider_name = "openai"
    
    async def generate(self, prompt: Union[str, List[Dict[str, str]]], max_tokens: int = 256, temperature: float = 0.1, **kwargs) -> LLMResponse:
        """
        Generate a response using OpenAI GPT models.
        Accepts either a string prompt or a list of message dicts with 'role' and 'content'.
        """
        if not OPENAI_AVAILABLE:
            raise ConnectionError("OpenAI package not installed. Install with: pip install openai")
            
        if not openai.api_key:
            raise ConnectionError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        try:
            from app.logging_config import log_llm_interaction, log_sensitive_debug, log_performance_metric
            
            gpt_start_time = time.time()
            
            # Convert to messages format if string prompt
            if isinstance(prompt, str):
                messages = [{"role": "user", "content": prompt}]
                total_prompt_tokens = estimate_tokens_from_text(prompt)
            else:
                messages = prompt
                # Estimate tokens from all messages
                total_prompt_tokens = sum(estimate_tokens_from_text(msg.get("content", "")) for msg in messages)
            
            log_llm_interaction(
                logger, "OPENAI_GPT", total_prompt_tokens, 0,  # response tokens unknown yet
                model=self.model_name, max_tokens=max_tokens, session_id=kwargs.get("session_id", "none")
            )
            
            log_sensitive_debug(
                logger, "GPT LLM request",
                messages=messages,
                model=self.model_name
            )
            
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            gpt_duration = (time.time() - gpt_start_time) * 1000
            response_tokens = response.usage.completion_tokens
            total_prompt_tokens = response.usage.prompt_tokens # More accurate from response
            
            log_llm_interaction(
                logger, "OPENAI_GPT", total_prompt_tokens, response_tokens,
                model=self.model_name, response_len=len(answer), 
                duration_ms=gpt_duration, session_id=kwargs.get("session_id", "none")
            )
            
            log_performance_metric(
                logger, "GPT_LLM_GENERATION", gpt_duration,
                model=self.model_name, prompt_tokens=total_prompt_tokens,
                response_tokens=response_tokens, session_id=kwargs.get("session_id")
            )
            
            log_sensitive_debug(
                logger, "GPT LLM response",
                response_text=answer, response_len=len(answer)
            )
            
            return LLMResponse(
                text=answer,
                metadata={
                    "model": self.model_name, 
                    "provider": self.provider_name,
                    "usage": response.usage
                },
                usage={
                    "prompt_tokens": total_prompt_tokens,
                    "completion_tokens": response_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                finish_reason=response.choices[0].finish_reason
            )
            
        except Exception as e:
            logger.exception("GPT API call failed: %s", e)
            raise ConnectionError(f"GPT API call failed: {e}")

    def get_provider_name(self) -> str:
        return self.provider_name
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def is_available(self) -> bool:
        return OPENAI_AVAILABLE and openai.api_key is not None
    
    def get_max_context_length(self) -> int:
        # This can be dynamic based on the model
        if "gpt-4" in self.model_name:
            return 8192
        return 4096
