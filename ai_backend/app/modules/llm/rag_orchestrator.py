"""RAG orchestration implementation."""

from typing import Dict, Any, List, Optional
import logging

from .interfaces import IRAGOrchestrator, ILLMProvider, IPromptManager, RAGRequest, RAGResponse
from .prompt_manager import OptimizedPromptManager
from ..vector_db.interfaces import IVectorStore
from ..auth.interfaces import IRBACManager

logger = logging.getLogger(__name__)


class RAGOrchestrator(IRAGOrchestrator):
    """RAG orchestration implementation."""
    
    def __init__(self, vector_store: IVectorStore, rbac_manager: IRBACManager, session_manager=None):
        self.vector_store = vector_store
        self.rbac_manager = rbac_manager
        self.session_manager = session_manager
        self.prompt_manager: IPromptManager = OptimizedPromptManager()
    
    async def process_query(self, request: RAGRequest) -> RAGResponse:
        """Process RAG query end-to-end."""
        try:
            # 1. Retrieve relevant documents
            documents = await self.retrieve_documents(
                request.question, 
                request.user, 
                request.top_k, 
                request.category
            )
            
            # 2. Build context from documents
            context = await self.build_context(documents)
            
            # 3. Get conversation history if session provided
            conversation_history = []
            if request.session_id and self.session_manager:
                conversation_history = await self.session_manager.get_messages(
                    request.session_id, limit=3
                )
            
            # 4. Build prompts
            system_prompt = await self.prompt_manager.build_system_prompt(
                request.user, context, request.category
            )
            user_prompt = await self.prompt_manager.build_user_prompt(
                request.question, context
            )
            final_prompt = await self.prompt_manager.build_full_prompt(
                system_prompt, user_prompt, conversation_history
            )
            
            # 5. Generate response if LLM requested
            answer = None
            if request.use_llm:
                # This would be injected with actual provider
                answer = f"Generated response for: {request.question}"
            
            # 6. Store message in session if provided
            if request.session_id and self.session_manager:
                await self.session_manager.store_message(
                    request.session_id, "user", request.question
                )
                if answer:
                    await self.session_manager.store_message(
                        request.session_id, "assistant", answer
                    )
            
            return RAGResponse(
                answer=answer,
                retrieved_documents=documents,
                context=context,
                final_prompt=final_prompt if request.debug else None,
                metadata={
                    "user_role": request.user.get("role"),
                    "documents_count": len(documents),
                    "session_id": request.session_id
                }
            )
            
        except Exception as e:
            logger.error(f"Error processing RAG query: {e}")
            return RAGResponse(
                answer=None,
                retrieved_documents=[],
                context="",
                metadata={"error": str(e)}
            )
    
    async def retrieve_documents(self, query: str, user: Dict[str, Any], top_k: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        try:
            # Build filter for category if provided
            filter_metadata = {}
            if category:
                filter_metadata["category"] = category
            
            # Search documents
            documents = await self.vector_store.search_documents(
                query, top_k * 2, filter_metadata  # Get more to allow for RBAC filtering
            )
            
            # Apply RBAC filtering
            filtered_documents = await self.rbac_manager.filter_documents(documents, user)
            
            # Return top_k after filtering
            return filtered_documents[:top_k]
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return []
    
    async def generate_response(self, prompt: str, provider: ILLMProvider, max_tokens: int = 256, temperature: float = 0.1) -> Optional[str]:
        """Generate response using LLM."""
        try:
            response = await provider.generate(
                prompt, max_tokens=max_tokens, temperature=temperature
            )
            return response.text
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
    
    async def build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context from retrieved documents."""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Unknown")
            
            context_parts.append(f"Document {i} (Source: {source}):\n{text}")
        
        return "\n\n".join(context_parts)


class MultiProviderRAGOrchestrator(RAGOrchestrator):
    """RAG orchestrator with multiple LLM provider support."""
    
    def __init__(self, vector_store: IVectorStore, rbac_manager: IRBACManager, session_manager=None):
        super().__init__(vector_store, rbac_manager, session_manager)
        self.providers: Dict[str, ILLMProvider] = {}
    
    def register_provider(self, name: str, provider: ILLMProvider):
        """Register an LLM provider."""
        self.providers[name] = provider
        logger.info(f"Registered LLM provider: {name}")
    
    async def process_query_with_provider(self, request: RAGRequest, provider_name: str) -> RAGResponse:
        """Process query with specific provider."""
        if provider_name not in self.providers:
            raise ValueError(f"Provider {provider_name} not registered")
        
        provider = self.providers[provider_name]
        
        # Process query normally
        response = await self.process_query(request)
        
        # Generate answer with specific provider if LLM requested
        if request.use_llm and response.final_prompt:
            answer = await self.generate_response(
                response.final_prompt, 
                provider, 
                request.max_tokens, 
                request.temperature
            )
            response.answer = answer
            
            # Update metadata
            response.metadata.update({
                "provider": provider_name,
                "model": provider.get_model_name()
            })
        
        return response
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers."""
        return [name for name, provider in self.providers.items() if provider.is_available()]