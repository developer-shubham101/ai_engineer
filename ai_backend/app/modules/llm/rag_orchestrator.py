"RAG orchestrator implementation."

import logging
from typing import List, Dict, Any, Optional

from .interfaces import IRAGOrchestrator, RAGRequest, RAGResponse, RetrievedDocument, LLMResponse

from .prompt_manager import PromptManager
from .prompt_chain import PromptChain
from .provider_factory import create_provider
from .middleware import create_default_middleware_stack

from ..vector_db.interfaces import IVectorStore
from ..auth.interfaces import ISessionManager

logger = logging.getLogger(__name__)


class RAGOrchestrator(IRAGOrchestrator):
    """RAG orchestrator implementation."""
    
    def __init__(self, vector_store: IVectorStore, session_manager: ISessionManager):
        self.vector_store = vector_store
        self.session_manager = session_manager
        self.prompt_manager = PromptManager()
        self.prompt_chain = PromptChain(session_manager)
        self.middleware_stack = create_default_middleware_stack()
    
    async def process_query(self, request: RAGRequest) -> RAGResponse:
        """Process RAG query end-to-end."""
        # Process request through middleware
        request = await self.middleware_stack.process_request(request)
        
        # Check for cached response as of now we don't need it but keep it for future
        if request.metadata and "_cached_response" in request.metadata:
            return await self.middleware_stack.process_response(request, request.metadata["_cached_response"])
        
        try:
            # 1. Enhance query using chain  as of now we don't need it but keep it for future
            enhanced_query = await self.prompt_chain.build_enhanced_query(
                question=request.question,
                user=request.user,
                session_id=request.session_id,
                category=request.category
            )
            
            # 2. Retrieve documents
            documents = await self.retrieve_documents(
                query=request.question,
                user=request.user,
                top_k=request.top_k,
                category=request.category
            )
            
            logger.info("Retrieved documents for query:\n %s", documents)

            # 2. Build context
            context = await self.build_context(documents)

            logger.info("Built context with %s", context)
            
            # 3. Generate response if LLM requested
            answer = None
            final_prompt = None
            if request.use_llm:
                logger.info("Generating response using LLM provider: %s", request.provider)    
                provider_config = request.provider_specific or {}
                provider = await create_provider(request.provider, provider_config)
                
                final_prompt = await self.prompt_chain.build_prompt(
                    question=request.question,
                    context=context,
                    user=request.user,
                    session_id=request.session_id,
                    category=request.category
                )
                logger.info("Generated final prompt: ====START==== \n%s\n====END===", final_prompt)
                
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
            
            response = RAGResponse(
                answer=answer,
                retrieved_documents=retrieved_docs,
                context=context,
                final_prompt=final_prompt,
                metadata={"provider": request.provider}
            )
            
            # Process response through middleware
            return await self.middleware_stack.process_response(request, response)
            
        except Exception as e:
            logger.error("RAG processing failed: %s", str(e), exc_info=True)
            error_response = RAGResponse(
                answer="Sorry, I encountered an error processing your request.",
                retrieved_documents=[],
                context="",
                metadata={"error": str(e)}
            )
            return await self.middleware_stack.process_response(request, error_response)
    
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
            logger.error("Document retrieval failed: %s", str(e), exc_info=True)
            return []
    
    async def generate_response(self, prompt: str, provider, max_tokens: int = 256, temperature: float = 0.1) -> LLMResponse | None: #-> Optional[str]:
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
            logger.error("LLM generation failed: %s", str(e), exc_info=True)
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
