"""ColabLLM RAG service implementation."""

import logging
from typing import Optional

from ..providers.colabllm import ColabLLMProvider

logger = logging.getLogger(__name__)


class ColabLLMRAGService:
    """RAG service using ColabLLM provider."""

    def __init__(self, vector_store, session_manager, conversation_manager=None, template_manager=None):
        self.vector_store = vector_store
        self.session_manager = session_manager
        self.conversation_manager = conversation_manager
        self.template_manager = template_manager
        self.provider_name = "colabllm"
        self.llm_provider = None

    def _get_llm_provider(self) -> ColabLLMProvider:
        """Get or create ColabLLM provider."""
        if not self.llm_provider:
            from ..config.settings import settings
            self.llm_provider = ColabLLMProvider(
                base_url=settings.COLABLLM_BASE_URL,
                api_key=settings.COLABLLM_API_KEY
            )
        return self.llm_provider

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
        """Generate response using ColabLLM provider."""
        if not use_llm:
            return None

        try:
            provider = self._get_llm_provider()

            # Build final prompt
            final_prompt = f"{final_prefix}\n\nContext:\n{context_text}\n\nQuestion: {query_text}"

            # Generate response
            response = await provider.generate(
                prompt=final_prompt,
                max_tokens=max_tokens,
                temperature=temperature
            )

            # Extract text from LLMResponse
            if hasattr(response, 'text'):
                return response.text
            else:
                return str(response)

        except Exception as e:
            logger.error(f"ColabLLM generation failed: {e}")
            return f"Error generating response: {str(e)}"
