"RAG orchestrator implementation."

import logging
from typing import List, Dict, Any, Optional

from .interfaces import IRAGOrchestrator, RAGRequest, RAGResponse, RetrievedDocument

from .prompt_manager import PromptManager
from .providers.providers import ProviderFactory

from ..vector_db.interfaces import IVectorStore
from ..auth.interfaces import ISessionManager

logger = logging.getLogger(__name__)


class RAGOrchestrator(IRAGOrchestrator):
    """RAG orchestrator implementation."""
    
    def __init__(self, vector_store: IVectorStore, session_manager: ISessionManager):
        self.vector_store = vector_store
        self.session_manager = session_manager
        self.prompt_manager = PromptManager()
    
    async def process_query(self, request: RAGRequest) -> RAGResponse:
        """Process RAG query end-to-end."""
        try:
            # 1. Retrieve documents
            documents = await self.retrieve_documents(
                query=request.question,
                user=request.user,
                top_k=request.top_k,
                category=request.category
            )
            
            # 2. Build context
            context = await self.build_context(documents)
            
            # 3. Generate response if LLM requested
            answer = None
            final_prompt = None
            if request.use_llm:
                provider = ProviderFactory.create_provider(request.provider, request.provider_specific.get("model_name") if request.provider_specific else None)
                
                system_prompt = await self.prompt_manager.build_system_prompt(request.user, context, request.category)
                user_prompt = await self.prompt_manager.build_user_prompt(request.question, context)
                final_prompt = await self.prompt_manager.build_full_prompt(system_prompt, user_prompt)
                
                response = await self.generate_response(final_prompt, provider, request.max_tokens, request.temperature)
                answer = response.text if response else "I found relevant documents but couldn't generate a response."

            # 4. Convert documents to response format
            retrieved_docs = [
                RetrievedDocument(
                    id=doc.get("id", "unknown"),
                    text=doc.get("text", ""),
                    metadata=doc.get("metadata", {}),
                    distance=doc.get("distance")
                )
                for doc in documents
            ]
            
            return RAGResponse(
                answer=answer,
                retrieved_documents=retrieved_docs,
                context=context,
                final_prompt=final_prompt,
                metadata={"provider": request.provider}
            )
            
        except Exception as e:
            logger.exception("RAG processing failed: %s", e)
            return RAGResponse(
                answer="Sorry, I encountered an error processing your request.",
                retrieved_documents=[],
                context="",
                metadata={"error": str(e)}
            )
    
    async def retrieve_documents(self, query: str, user: Dict[str, Any], top_k: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        try:
            # Use vector store to search
            results = await self.vector_store.search_documents(
                query=query,
                top_k=top_k,
                metadata_filter={"category": category} if category else None
            )
            
            # Apply RBAC filtering (simplified)
            filtered_results = []
            user_role = user.get("role", "Guest")
            user_dept = user.get("department", "General")
            
            for doc in results:
                metadata = doc.get("metadata", {})
                
                # Simple RBAC check
                doc_dept = metadata.get("department", "General")
                sensitivity = metadata.get("sensitivity", "public_internal")
                
                # Allow if public or same department
                if sensitivity == "public_internal" or doc_dept == user_dept or user_role in ["SuperAdmin", "Manager"]:
                    filtered_results.append(doc)
            
            return filtered_results
            
        except Exception as e:
            logger.exception("Document retrieval failed: %s", e)
            return []
    
    async def generate_response(self, prompt: str, provider, max_tokens: int = 256, temperature: float = 0.1) : #-> Optional[str]:
        """Generate response using LLM."""
        try:
            if provider and hasattr(provider, 'generate'):
                response = await provider.generate(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response
            return None
        except Exception as e:
            logger.exception("LLM generation failed: %s", e)
            return None
    
    async def build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context from retrieved documents."""
        if not documents:
            return ""
        
        context_parts = []
        for i, doc in enumerate(documents[:5]):  # Limit to top 5
            text = doc.get("text", "")
            if text:
                context_parts.append(f"Document {i+1}: {text[:500]}...")  # Truncate
        
        return "\n\n".join(context_parts)
