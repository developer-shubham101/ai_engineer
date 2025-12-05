"RAG orchestrator implementation."

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

from .interfaces import IRAGOrchestrator, RAGRequest, RAGResponse, RetrievedDocument, LLMResponse

from .prompt_manager import PromptManager
from .prompt_chain import PromptChain
from .langchain_prompt_selector import ConditionalPromptSelector
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
        self.langchain_selector = ConditionalPromptSelector()
        self.middleware_stack = create_default_middleware_stack()
    
    async def process_query(self, request: RAGRequest) -> RAGResponse:
        """Process RAG query with three flows:
        1. Authenticated Company User (non-Guest): Profile from user_meta, session tracking
        2. Authenticated Guest User: Onboarding questions, session tracking
        3. Unauthenticated User: Basic RAG without session
        """
        request = await self.middleware_stack.process_request(request)
        
        if request.metadata and "_cached_response" in request.metadata:
            return await self.middleware_stack.process_response(request, request.metadata["_cached_response"])
        
        try:
            user_id = request.user.get("user_id") if request.user else None
            user_role = request.user.get("role") if request.user else None
            session_id = request.session_id
            
            # CASE 1: AUTHENTICATED COMPANY USER (non-Guest)
            if user_id and user_role and user_role != "Guest":
                profile = await self._handle_company_user(request.user, session_id)
                session_history = await self._get_session_history(session_id)
                
            # CASE 2: AUTHENTICATED GUEST USER
            elif user_id and user_role == "Guest":
                onboarding_response = await self._handle_guest_onboarding(request.question, session_id)
                logger.info("Onboarding response: %s", onboarding_response)
                if onboarding_response:
                    return await self.middleware_stack.process_response(request, onboarding_response)
                profile = await self._get_guest_profile(session_id)
                session_history = await self._get_session_history(session_id)
                
            # CASE 3: UNAUTHENTICATED USER
            else:
                profile = {}
                session_history = []
                session_id = None
            
            log_timestamp = datetime.now().isoformat()
            logger.info(f"""
            [DEBUG LOG - {log_timestamp}]
            ==================================================
            User Query: {request.question}
            User Profile: {profile}
            ==================================================
            """)

            # Update request with profile and history
            if request.user:
                request.user.update(profile)
            
            # Retrieve documents only if use_documents is True
            documents = []
            if request.use_documents:
                documents = await self.retrieve_documents(
                    query=request.question,
                    user=request.user or {},
                    top_k=request.top_k,
                    category=request.category
                )
                log_timestamp = datetime.now().isoformat()
                logger.info(f"""
                [DEBUG LOG - {log_timestamp}]
                ==================================================
                Fetched Documents: {documents}
                ==================================================
                """)
            else:
                logger.info("Document retrieval skipped (use_documents=False)")

            # 2. Build context
            context = await self.build_context(documents)
            
            # Generate response if LLM requested
            answer = None
            final_prompt = None
            if request.use_llm:
                logger.info("Generating response using LLM provider: %s", request.provider)    
                provider_config = request.provider_specific or {}
                provider = await create_provider(request.provider, provider_config)
                
                # Format history
                history_str = ""
                if session_history:
                    history_str = self.session_manager.render_history(session_history)

                USE_CUSTOM_PROMPT_BUILDER = False
                if USE_CUSTOM_PROMPT_BUILDER :
                    final_prompt = await self.prompt_chain.build_prompt(
                        question=request.question,
                        context=context,
                        user=request.user,
                        session_id=session_id,
                        history=history_str,
                        category=request.category
                    )
                else:
                    # Use LangChain dynamic prompt selector
                    user_role = request.user.get("role", "Guest") if request.user else "Guest"
                    department = request.user.get("department", "General") if request.user else "General"

                    # Get context size from provider
                    context_size = getattr(provider, 'context_size', 2048) if provider else 2048

                    # Format source docs
                    source_docs = "\n".join([f"[{doc.get('id', 'unknown')}]: {doc.get('text', '')[:200]}..."
                                            for doc in documents[:5]])

                    # Prepare prompt data
                    prompt_data = {
                        "context_size": context_size,
                        "user_role": user_role,
                        "department": department,
                        "user_question": request.question,
                        "max_tokens": request.max_tokens,
                        "user_profile_summary": str(profile) if profile else "No profile",
                        "source_docs": source_docs,
                        "history": history_str  # Include history in prompt_data
                    }

                    # Create dynamic prompt
                    template = self.langchain_selector.get_prompt_template(prompt_data)

                    if not template:
                        logger.error("Failed to get prompt template.")
                        final_prompt = f"Question: {request.question}\nSources: {source_docs}"
                    else:
                        final_prompt = self.langchain_selector.format_prompt(
                            template=template,
                            prompt_data=prompt_data
                        )
                logger.info("Generated final prompt: ====START==== \n%s\n====END===", final_prompt)
                
                response = await self.generate_response(final_prompt, provider, request.max_tokens, request.temperature)
                
                # Validate JSON response
                if response and response.text:
                    validated = self.langchain_selector.validate_response(response.text)
                    if validated:
                        answer = validated.model_dump_json()
                    else:
                        # Retry with fallback template
                        fallback_template = self.langchain_selector.get_fallback_template()
                        fallback_prompt = fallback_template.format(
                            user_question=prompt_data.get("user_question", ""),
                            source_docs=prompt_data.get("source_docs", "")
                        )
                        retry_response = await self.generate_response(fallback_prompt, provider, request.max_tokens, request.temperature)
                        answer = retry_response.text if retry_response else '{"answer": "I could not process your request", "sources": [], "confidence": "low"}'
                else:
                    answer = '{"answer": "I found relevant documents but couldn\'t generate a response", "sources": [], "confidence": "low"}'
                
                log_timestamp = datetime.now().isoformat()
                logger.info(f"""
                [DEBUG LOG - {log_timestamp}]
                ==================================================
                Final AI Response: {answer}
                ==================================================
                """)

                # Store conversation if session exists
                if session_id:
                    await self._store_conversation(session_id, request.question, answer)

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
    
    async def _handle_company_user(self, user: Dict[str, Any], session_id: Optional[str]) -> Dict[str, Any]:
        """Handle authenticated company user flow."""
        user_id = user.get("user_id")
        profile = {}
        
        # Get profile from user_meta
        if user_id:
            user_manager = self.session_manager  # Assuming session_manager has user methods
            if hasattr(user_manager, 'get_all_user_meta'):
                profile = user_manager.get_all_user_meta(user_id) or {}
        
        # Ensure session exists
        if session_id and hasattr(self.session_manager, 'session_exists'):
            if not self.session_manager.session_exists(session_id):
                self.session_manager.create_session(
                    session_id=session_id,
                    role=user.get("role"),
                    department=user.get("department")
                )
        
        return profile
    
    async def _handle_guest_onboarding(self, question: str, session_id: Optional[str]) -> Optional[RAGResponse]:
        """Handle guest user onboarding flow."""
        if not session_id or not hasattr(self.session_manager, 'get_next_missing_profile_key'):
            return None
        
        # Ensure guest session exists
        if not self.session_manager.session_exists(session_id):
            self.session_manager.create_session(
                session_id=session_id,
                role="Guest",
                department="General"
            )
        
        # Check for onboarding
        next_field = self.session_manager.get_next_missing_profile_key(session_id)
        logger.info("Next missing profile key: %s", next_field)
        if not next_field:
            return None
        
        # Get recent messages
        session_history = self.session_manager.fetch_recent_messages(session_id, limit=3)
        logger.info("Session history: %s", session_history)
        
        # Check if this is a response to onboarding question
        if session_history:
            last_msg = session_history[-1]
            if (last_msg.get("speaker") == "assistant" and 
                last_msg.get("content", "").strip() == next_field["question"].strip()):
                
                # Save user response
                self.session_manager.store_message(session_id, "user", question)
                self.session_manager.set_profile_value(session_id, next_field["key"], question.strip())
                
                # Check for next question
                next_field = self.session_manager.get_next_missing_profile_key(session_id)
                if next_field:
                    self.session_manager.store_message(session_id, "assistant", next_field["question"])
                    return RAGResponse(answer=next_field["question"], retrieved_documents=[], context="")
                else:
                    completion_msg = "Thank you! Your details have been saved."
                    self.session_manager.store_message(session_id, "assistant", completion_msg)
                    return RAGResponse(answer=completion_msg, retrieved_documents=[], context="")
        
        # Ask first onboarding question
        self.session_manager.store_message(session_id, "assistant", next_field["question"])
        return RAGResponse(answer=next_field["question"], retrieved_documents=[], context="")
    
    async def _get_guest_profile(self, session_id: Optional[str]) -> Dict[str, Any]:
        """Get guest profile from session."""
        if not session_id or not hasattr(self.session_manager, 'get_full_profile'):
            return {}
        return self.session_manager.get_full_profile(session_id) or {}
    
    async def _get_session_history(self, session_id: Optional[str]) -> List[Dict[str, Any]]:
        """Get session conversation history."""
        if not session_id or not hasattr(self.session_manager, 'fetch_recent_messages'):
            return []
        return self.session_manager.fetch_recent_messages(session_id, limit=5) or []
    
    async def _store_conversation(self, session_id: str, question: str, answer: str):
        """Store conversation in session."""
        if hasattr(self.session_manager, 'store_message'):
            self.session_manager.store_message(session_id, "user", question)
            if answer:
                self.session_manager.store_message(session_id, "assistant", answer)
    
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
