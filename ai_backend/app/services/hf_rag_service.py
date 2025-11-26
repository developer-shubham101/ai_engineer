# app/services/hf_rag_service.py
"""
Hugging Face RAG service implementation using HF Inference API.
Example implementation showing how to extend BaseRAGService for new providers.
"""

import logging
from typing import Optional, Dict, Any
import os

from app.services.base_rag_service import BaseRAGService
from app.services.prompt_builder import build_prompt_with_selected_chunks

logger = logging.getLogger(__name__)

# Optional Hugging Face import
try:
    from huggingface_hub import InferenceClient
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.warning("Hugging Face Hub not available. Install huggingface_hub package to use HF RAG service.")


class HuggingFaceRAGService(BaseRAGService):
    """
    Hugging Face RAG service implementation using HF Inference API.
    Inherits common functionality from BaseRAGService.
    """
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        super().__init__()
        self.model_name = model_name
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        
        if HF_AVAILABLE and self.hf_token:
            self.client = InferenceClient(token=self.hf_token)
        else:
            self.client = None
    
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
        Generate a response using Hugging Face Inference API.
        """
        if not use_llm:
            return None
            
        if not HF_AVAILABLE:
            raise ConnectionError("Hugging Face Hub not installed. Install with: pip install huggingface_hub")
            
        if not self.client:
            raise ConnectionError("Hugging Face API token not found. Set HUGGINGFACE_API_TOKEN environment variable.")

        prompt = build_prompt_with_selected_chunks(final_prefix, context_text, query_text)

        try:
            response = self.client.text_generation(
                prompt=prompt,
                model=self.model_name,
                max_new_tokens=max_tokens,
                temperature=0.1,
                return_full_text=False
            )
            
            answer = response if isinstance(response, str) else response.get("generated_text", "")
            logger.info("HF returned answer (length=%d) for session=%s", len(answer), session_id)
            return answer
            
        except Exception as e:
            logger.exception("Hugging Face API call failed: %s", e)
            raise


# Create global instance
_hf_rag_service = HuggingFaceRAGService()


async def query_hf_rag(
        query_text: str,
        n_results: int = 3,
        requester: Optional[Dict[str, str]] = None,
        llm_prompt_prefix: Optional[str] = None,
        use_llm: bool = True,
        max_tokens: int = 256,
        session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query the Hugging Face RAG service using the base RAG functionality.
    """
    return await _hf_rag_service.query_rag(
        query_text=query_text,
        n_results=n_results,
        requester=requester,
        llm_prompt_prefix=llm_prompt_prefix,
        use_llm=use_llm,
        max_tokens=max_tokens,
        session_id=session_id
    )