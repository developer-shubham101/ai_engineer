# app/services/gpt_rag_service.py
"""
GPT RAG service implementation using OpenAI models.
Example implementation showing how to extend BaseRAGService for new providers.
"""

import logging
from typing import Optional, Dict, Any
import os

from app.services.base_rag_service import BaseRAGService
from app.services.prompt_builder import build_prompt_with_selected_chunks

logger = logging.getLogger(__name__)

# Optional OpenAI import
try:
    import openai
    openai.api_key = os.getenv("OPENAI_API_KEY")
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("OpenAI not available. Install openai package to use GPT RAG service.")


class GPTRAGService(BaseRAGService):
    """
    GPT RAG service implementation using OpenAI models.
    Inherits common functionality from BaseRAGService.
    """
    
    def __init__(self, model_name: str = "gpt-3.5-turbo"):
        super().__init__()
        self.model_name = model_name
    
    async def generate_response(
        self,
        query_text: str,
        context_text: str,
        final_prefix: str,
        use_llm: bool,
        max_tokens: int,
        temperature: float,
        session_id: Optional[str]
    ) -> Optional[str]:
        """
        Generate a response using OpenAI GPT models.
        """
        if not use_llm:
            return None
            
        if not OPENAI_AVAILABLE:
            raise ConnectionError("OpenAI package not installed. Install with: pip install openai")
            
        if not openai.api_key:
            raise ConnectionError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        prompt = build_prompt_with_selected_chunks(final_prefix, context_text, query_text)

        try:
            from app.logging_config import log_llm_interaction, log_sensitive_debug, log_performance_metric
            from app.services.prompt_builder import estimate_tokens_from_text
            import time
            
            gpt_start_time = time.time()
            
            # Estimate tokens for logging
            system_tokens = estimate_tokens_from_text(final_prefix)
            user_content = f"Context:\n{context_text}\n\nQuestion: {query_text}"
            user_tokens = estimate_tokens_from_text(user_content)
            total_prompt_tokens = system_tokens + user_tokens
            
            log_llm_interaction(
                logger, "OPENAI_GPT", total_prompt_tokens, 0,  # response tokens unknown yet
                model=self.model_name, max_tokens=max_tokens, session_id=session_id or "none"
            )
            
            log_sensitive_debug(
                logger, "GPT LLM request",
                system_content=final_prefix, user_content=user_content,
                model=self.model_name
            )
            
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": final_prefix},
                    {"role": "user", "content": user_content}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            
            answer = response.choices[0].message.content
            gpt_duration = (time.time() - gpt_start_time) * 1000
            response_tokens = estimate_tokens_from_text(answer)
            
            log_llm_interaction(
                logger, "OPENAI_GPT", total_prompt_tokens, response_tokens,
                model=self.model_name, response_len=len(answer), 
                duration_ms=gpt_duration, session_id=session_id or "none"
            )
            
            log_performance_metric(
                logger, "GPT_LLM_GENERATION", gpt_duration,
                model=self.model_name, prompt_tokens=total_prompt_tokens,
                response_tokens=response_tokens, session_id=session_id
            )
            
            log_sensitive_debug(
                logger, "GPT LLM response",
                response_text=answer, response_len=len(answer)
            )
            
            return answer
            
        except Exception as e:
            logger.exception("GPT API call failed: %s", e)
            raise


# Create global instance
_gpt_rag_service = GPTRAGService()


async def query_gpt_rag(
        query_text: str,
        n_results: int = 3,
        requester: Optional[Dict[str, str]] = None,
        llm_prompt_prefix: Optional[str] = None,
        use_llm: bool = True,
        max_tokens: int = 256,
        temperature: float = 0.1,
        session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query the GPT RAG service using the base RAG functionality.
    """
    return await _gpt_rag_service.query_rag(
        query_text=query_text,
        n_results=n_results,
        requester=requester,
        llm_prompt_prefix=llm_prompt_prefix,
        use_llm=use_llm,
        max_tokens=max_tokens,
        temperature=temperature,
        session_id=session_id
    )