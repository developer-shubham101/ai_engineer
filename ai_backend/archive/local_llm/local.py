# app/modules/llm/providers/local.py
"""
Local LLM provider implementation.
"""
from __future__ import annotations

import logging
from typing import Optional, Dict, Any, Union, List
import time

from app.modules.llm.interfaces import ILLMProvider, LLMResponse
from app.modules.llm.prompt_builder import (
    estimate_tokens_from_text,
    build_prompt_with_selected_chunks,
)
from app.modules.llm.model_manager import get_llm_instance
from app.modules.llm.prompt_builder import _call_llm_with_retry

logger = logging.getLogger(__name__)


class LocalLLMProvider(ILLMProvider):
    """
    Local LLM provider implementation.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name
        self.provider_name = "local"
        self._llm_instance = None

    def _get_llm_instance(self):
        if not self._llm_instance:
            self._llm_instance = get_llm_instance(self.model_name)
        return self._llm_instance
    
    async def generate(self, prompt: Union[str, List[Dict[str, str]]], max_tokens: int = 256, temperature: float = 0.1, **kwargs) -> LLMResponse:
        """
        Generate a response using local LLM.
        Accepts either a string prompt or a list of message dicts with 'role' and 'content'.
        """
        model_key = kwargs.get("model_key")
        if model_key:
            self.model_name = model_key
            self._llm_instance = None # Reset instance to load new model
        
        # Convert messages to prompt if needed
        if isinstance(prompt, list):
            logger.debug("Converting messages to prompt for local LLM")
            prompt = self._messages_to_prompt(prompt)
        
        logger.debug("Generating response with local LLM, model_key=%s", self.model_name)
        
        try:
            llm_instance = self._get_llm_instance()
        except Exception as e:
            logger.exception("Failed to load LLM instance: %s", e)
            raise ConnectionError(f"Failed to load LLM instance: {e}")

        try:
            # Calculate prompt metrics
            prompt_tokens = estimate_tokens_from_text(prompt)

            from app.logging_config import log_llm_interaction, log_performance_metric, log_sensitive_debug
            
            llm_start_time = time.time()

            log_llm_interaction(
                logger, "LOCAL_LLM", prompt_tokens, 0,  # response tokens unknown yet
                prompt_len=len(prompt), max_tokens=max_tokens,
                session_id=kwargs.get("session_id", "none")
            )

            log_sensitive_debug(
                logger, "Local LLM full prompt",
                full_prompt=prompt, prompt_len=len(prompt), prompt_tokens=prompt_tokens
            )

            # Check if prompt might exceed context window
            estimated_total = prompt_tokens + max_tokens
            context_window_limit = self.get_max_context_length()
            if estimated_total > context_window_limit:
                from app.logging_config import log_security_event
                log_security_event(
                    logger, "POTENTIAL_CONTEXT_OVERFLOW", "system",
                    estimated_total=estimated_total, prompt_tokens=prompt_tokens,
                    max_tokens=max_tokens, context_limit=context_window_limit,
                    session_id=kwargs.get("session_id")
                )

            answer = await _call_llm_with_retry(
                llm_instance,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            response_len = len(answer or "")
            response_tokens = estimate_tokens_from_text(answer or "")
            llm_duration = (time.time() - llm_start_time) * 1000

            log_llm_interaction(
                logger, "LOCAL_LLM", prompt_tokens, response_tokens,
                response_len=response_len,
                duration_ms=llm_duration, session_id=kwargs.get("session_id", "none")
            )

            log_performance_metric(
                logger, "LOCAL_LLM_GENERATION", llm_duration,
                prompt_tokens=prompt_tokens,
                response_tokens=response_tokens, session_id=kwargs.get("session_id")
            )

            log_sensitive_debug(
                logger, "Local LLM response",
                response_text=answer or "", response_len=response_len,
                response_tokens=response_tokens
            )
            
            return LLMResponse(
                text=answer or "",
                metadata={"model": self.model_name or "default", "provider": self.provider_name},
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": response_tokens,
                    "total_tokens": prompt_tokens + response_tokens
                },
                finish_reason="completed"
            )

        except Exception as e:
            logger.exception("LLM call failed: %s", e)
            raise ConnectionError(f"LLM call failed: {e}")

    def get_provider_name(self) -> str:
        return self.provider_name

    def get_model_name(self) -> str:
        if self.model_name:
            return self.model_name
        
        # If no model name is set, we need a way to get the default.
        # This will be handled by the model manager.
        from app.modules.llm.model_manager import get_model_manager
        manager = get_model_manager()
        return manager.get_default_model()

    def is_available(self) -> bool:
        from app.modules.llm.model_manager import get_model_manager
        manager = get_model_manager()
        best_model = manager.get_best_available_model()
        return best_model is not None

    def get_max_context_length(self) -> int:
        # A common default for local models
        return 2048
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages array to single prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        return "\n\n".join(prompt_parts)
