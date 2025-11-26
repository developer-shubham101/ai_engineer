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
            response = await openai.ChatCompletion.acreate(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": final_prefix},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query_text}"}
                ],
                max_tokens=max_tokens,
                temperature=0.0
            )
            
            answer = response.choices[0].message.content
            logger.info("GPT returned answer (length=%d) for session=%s", len(answer), session_id)
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
        session_id=session_id
    )