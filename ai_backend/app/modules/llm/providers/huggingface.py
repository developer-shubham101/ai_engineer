# app/modules/llm/providers/huggingface.py
"""
Hugging Face LLM provider implementation using HF Inference API.
"""

import logging
from typing import Optional, Dict, Any, Union, List
import os
import time

from app.modules.llm.interfaces import ILLMProvider, LLMResponse
from app.modules.llm.prompt_builder import build_prompt_with_selected_chunks, estimate_tokens_from_text

logger = logging.getLogger(__name__)

# Optional Hugging Face import
try:
    from huggingface_hub import InferenceClient, HfFolder
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("Hugging Face Hub not available. Install huggingface_hub package to use HF RAG service.")


class HuggingFaceLLMProvider(ILLMProvider):
    """
    Hugging Face LLM provider implementation using HF Inference API.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.provider_name = "huggingface"
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN") or HfFolder.get_token()
        
        if HF_AVAILABLE and self.hf_token:
            self.client = InferenceClient(token=self.hf_token)
        else:
            self.client = None
    
    async def generate(self, prompt: Union[str, List[Dict[str, str]]], max_tokens: int = 256, temperature: float = 0.1, **kwargs) -> LLMResponse:
        """
        Generate a response using Hugging Face Inference API.
        Accepts either a string prompt or a list of message dicts with 'role' and 'content'.
        """
        if not HF_AVAILABLE:
            raise ConnectionError("Hugging Face Hub not installed. Install with: pip install huggingface_hub")
            
        if not self.client:
            raise ConnectionError("Hugging Face API token not found. Set HUGGINGFACE_API_TOKEN environment variable.")

        try:
            from app.logging_config import log_llm_interaction, log_sensitive_debug, log_performance_metric
            
            hf_start_time = time.time()
            
            # Convert messages to prompt if needed
            if isinstance(prompt, list):
                logger.debug("Converting messages to prompt for Hugging Face LLM")
                prompt_str = self._messages_to_prompt(prompt)
            else:
                prompt_str = prompt
            
            prompt_tokens = estimate_tokens_from_text(prompt_str)
            
            log_llm_interaction(
                logger, "HUGGINGFACE", prompt_tokens, 0,
                model=self.model_name, max_tokens=max_tokens, session_id=kwargs.get("session_id", "none")
            )
            
            log_sensitive_debug(
                logger, "Hugging Face LLM request",
                prompt=prompt_str, model=self.model_name
            )

            # The text_generation method is not async, so we'd need to run it in a threadpool
            from fastapi.concurrency import run_in_threadpool
            response = await run_in_threadpool(
                self.client.text_generation,
                prompt=prompt_str,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else None, # Some models fail with temp=0
                return_full_text=False
            )
            
            answer = response if isinstance(response, str) else ""
            hf_duration = (time.time() - hf_start_time) * 1000
            response_tokens = estimate_tokens_from_text(answer)
            
            log_llm_interaction(
                logger, "HUGGINGFACE", prompt_tokens, response_tokens,
                model=self.model_name, response_len=len(answer), 
                duration_ms=hf_duration, session_id=kwargs.get("session_id", "none")
            )
            
            log_performance_metric(
                logger, "HF_LLM_GENERATION", hf_duration,
                model=self.model_name, prompt_tokens=prompt_tokens,
                response_tokens=response_tokens, session_id=kwargs.get("session_id")
            )
            
            log_sensitive_debug(
                logger, "Hugging Face LLM response",
                response_text=answer, response_len=len(answer)
            )
            
            return LLMResponse(
                text=answer,
                metadata={"model": self.model_name, "provider": self.provider_name},
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": response_tokens,
                    "total_tokens": prompt_tokens + response_tokens
                },
                finish_reason="completed"
            )
            
        except Exception as e:
            logger.exception("Hugging Face API call failed: %s", e)
            raise ConnectionError(f"Hugging Face API call failed: {e}")

    def get_provider_name(self) -> str:
        return self.provider_name
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def is_available(self) -> bool:
        return self.client is not None
    
    def get_max_context_length(self) -> int:
        # This is highly model-dependent. A default, but should be looked up.
        return 2048
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages array to single prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        return "\n\n".join(prompt_parts)
