import os
import logging
import time
from typing import Optional, Dict, Any, Union, List

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from fastapi.concurrency import run_in_threadpool

from app.modules.llm.interfaces import ILLMProvider, LLMResponse
from app.modules.llm.prompt_builder import estimate_tokens_from_text # Temporarily still using this

load_dotenv()
logger = logging.getLogger(__name__)

google_ai_models = [
    "gemini-nano",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
]

class GoogleLLMProvider(ILLMProvider):
    """
    Google LLM provider implementation using Google Gemini models.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-flash-lite"):
        self.model_name = model_name
        self.provider_name = "google"
        self.google_api_key = os.environ.get("GOOGLE_API_KEY")
        self.llm = self._initialize_llm()

    def _initialize_llm(self):
        if not self.google_api_key:
            logger.warning("GOOGLE_API_KEY not found. GoogleLLMProvider will not be available.")
            return None
        try:
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.google_api_key,
                convert_system_message_to_human=True
            )
        except Exception as e:
            logger.error(f"Failed to initialize Google LLM: {e}")
            return None

    async def generate(self, prompt: Union[str, List[Dict[str, str]]], max_tokens: int = 256, temperature: float = 0.1, **kwargs) -> LLMResponse:
        """
        Generate a response using Google Gemini LLM.
        Accepts either a string prompt or a list of message dicts with 'role' and 'content'.
        """
        if not self.llm:
            raise ConnectionError("Google LLM is not initialized. Check your API key.")

        try:
            from app.logging_config import log_llm_interaction, log_sensitive_debug, log_performance_metric
            
            google_start_time = time.time()
            
            # Convert messages to prompt if needed
            if isinstance(prompt, list):
                logger.debug("Converting messages to prompt for Google LLM")
                prompt_str = self._messages_to_prompt(prompt)
            else:
                prompt_str = prompt
            
            prompt_tokens = estimate_tokens_from_text(prompt_str)
            
            log_llm_interaction(
                logger, "GOOGLE_GEMINI", prompt_tokens, 0,  # response tokens unknown yet
                prompt_len=len(prompt_str), session_id=kwargs.get("session_id", "none"),
                model=self.model_name
            )
            
            log_sensitive_debug(
                logger, "Google LLM request",
                full_prompt=prompt_str, prompt_len=len(prompt_str)
            )
            
            temp_llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.google_api_key,
                convert_system_message_to_human=True,
                temperature=temperature
            )
            answer = await run_in_threadpool(temp_llm.invoke, prompt_str)
            answer_content = answer.content if answer and hasattr(answer, 'content') else str(answer)
            
            google_duration = (time.time() - google_start_time) * 1000
            response_tokens = estimate_tokens_from_text(answer_content)
            
            log_llm_interaction(
                logger, "GOOGLE_GEMINI", prompt_tokens, response_tokens,
                response_len=len(answer_content), duration_ms=google_duration,
                session_id=kwargs.get("session_id", "none"), model=self.model_name
            )
            
            log_performance_metric(
                logger, "GOOGLE_LLM_GENERATION", google_duration,
                prompt_tokens=prompt_tokens, response_tokens=response_tokens,
                session_id=kwargs.get("session_id")
            )
            
            log_sensitive_debug(
                logger, "Google LLM response",
                response_text=answer_content, response_len=len(answer_content)
            )
            
            return LLMResponse(
                text=answer_content,
                metadata={"model": self.model_name, "provider": self.provider_name},
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": response_tokens,
                    "total_tokens": prompt_tokens + response_tokens
                },
                finish_reason="completed"
            )
        except Exception as e:
            logger.exception("Google LLM call failed: %s", e)
            raise ConnectionError(f"Google LLM call failed: {e}")

    def get_provider_name(self) -> str:
        return self.provider_name
    
    def get_model_name(self) -> str:
        return self.model_name
    
    def is_available(self) -> bool:
        return self.llm is not None
    
    def get_max_context_length(self) -> int:
        return 32768
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages array to single prompt string."""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            prompt_parts.append(f"{role}: {content}")
        return "\n\n".join(prompt_parts)
