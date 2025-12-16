"RAG orchestrator implementation."

import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional

from .interfaces import IRAGOrchestrator, RAGRequest, RAGResponse, RetrievedDocument, LLMResponse
from .langchain_prompt_selector import ConditionalPromptSelector
from .middleware import create_default_middleware_stack
from .prompt_chain import PromptChain
from .prompt_manager import PromptManager
from .provider_factory import create_provider
from ..auth.interfaces import ISessionManager
from ..vector_db.interfaces import IVectorStore

logger = logging.getLogger(__name__)


class RAGOrchestrator(IRAGOrchestrator):
    """RAG orchestrator implementation."""

    def __init__(self, vector_store: IVectorStore, session_manager: ISessionManager, conversation_manager=None, template_manager=None):
        self.vector_store = vector_store
        self.session_manager = session_manager
        self.conversation_manager = conversation_manager  # NEW: Conversation manager for persistent history
        self.prompt_manager = PromptManager()
        self.prompt_chain = PromptChain(session_manager)
        self.langchain_selector = ConditionalPromptSelector(template_manager)
        self.middleware_stack = create_default_middleware_stack()

    async def process_query(self, request: RAGRequest) -> RAGResponse:
        """Process RAG query with three flows:
        1. Authenticated Company User (non-Guest): Profile from user_meta, session tracking
        2. Authenticated Guest User: Onboarding questions, session tracking
        3. Unauthenticated User: Basic RAG without session
        """
        global prompt_data
        request = await self.middleware_stack.process_request(request)

        start_time = time.time()

        if request.metadata and "_cached_response" in request.metadata:
            return await self.middleware_stack.process_response(request, request.metadata["_cached_response"])

        try:
            user_id = request.user.get("user_id") if request.user else None
            user_role = request.user.get("role") if request.user else None
            session_id = request.session_id

            # Get conversation_id from session if available
            conversation_id = request.conversation_id
            logger.info(f"Conversation ID: {conversation_id}")

            if session_id and hasattr(self.session_manager, 'get_session'):
                try:
                    session_data = self.session_manager.get_session(session_id)
                    # conversation_id = session_data.get("conversation_id") if session_data else None
                except Exception as e:
                    logger.debug(f"Could not get conversation_id from session: {e}")

            # CASE 1: AUTHENTICATED COMPANY USER (non-Guest)
            if user_id and user_role and user_role != "Guest":
                profile = await self._handle_company_user(request.user, session_id)
                session_history = await self._get_session_history(session_id, user_id, conversation_id)

            # CASE 2: AUTHENTICATED GUEST USER
            elif user_id and user_role == "Guest":
                onboarding_response = await self._handle_guest_onboarding(request.question, session_id, conversation_id,
                                                                          user_id)
                logger.info("Onboarding response: %s", onboarding_response)
                if onboarding_response:
                    return await self.middleware_stack.process_response(request, onboarding_response)
                profile = await self._get_guest_profile(session_id)
                session_history = await self._get_session_history(session_id, user_id, conversation_id)

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
                # log_timestamp = datetime.now().isoformat()
                # logger.info(f"""
                # [DEBUG LOG - {log_timestamp}]
                # ==================================================
                # Fetched Documents: {documents}
                # ==================================================
                # """)
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
                if USE_CUSTOM_PROMPT_BUILDER:
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
                    source_docs = "\n".join(
                        [doc.get("text", "") for doc in documents[:5]]
                    )

                    logger.info("Source docs: \n%s\n=======\n", source_docs)
                    logger.info("History: \n%s\n=======\n", history_str)

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

                    logger.info("Prompt data: %s", prompt_data) 
                    logger.info("Prompt template: %s", request.prompt_template)    
                    # Create dynamic prompt
                    final_prompt = self.langchain_selector.format_prompt(prompt_data, request.prompt_template)

                    # if not template:
                    #     logger.error("Failed to get prompt template.")
                    #     final_prompt = f"Question: {request.question}\nSources: {source_docs}"
                    # else:
                    #     final_prompt = self.langchain_selector.format_prompt(
                    #         template=template,
                    #         prompt_data=prompt_data
                    #     )
                logger.info("Generated final prompt: ====START==== \n%s\n====END===", final_prompt)

                response = await self.generate_response(final_prompt, provider, request.max_tokens, request.temperature)

                # Validate JSON response
                if response and response.text:
                    logger.info("Generated response: ====START==== \n%s\n====END===", response.text)
                    answer = response.text
                    # validated = self.langchain_selector.validate_response(response.text)
                    # if validated:
                    #     logger.info("Validated response: ====START==== \n%s\n====END===", validated.model_dump_json())
                    #     answer = validated.model_dump_json()
                    # else:
                    #     logger.info("Failed to validate response: ====START==== \n%s\n====END===", response.text)
                    #     # Retry with fallback template
                    #     fallback_template = self.langchain_selector.get_fallback_template()
                    #     fallback_prompt = fallback_template.format(
                    #         user_question=prompt_data.get("user_question", ""),
                    #         source_docs=prompt_data.get("source_docs", "")
                    #     )
                    #     retry_response = await self.generate_response(fallback_prompt, provider, request.max_tokens,
                    #                                                   request.temperature)
                    #     answer = retry_response.text if retry_response else '{"answer": "I could not process your request", "sources": [], "confidence": "low"}'
                else:
                    answer = '{"answer": "I found relevant documents but couldn\'t generate a response", "sources": [], "confidence": "low"}'

                log_timestamp = datetime.now().isoformat()
                logger.info(f"""
                [DEBUG LOG - {log_timestamp}]
                ==================================================
                Final AI Response: {answer}
                ==================================================
                """)

                logger.info("Storing conversation with RAG logging in conversation {conversation_id}")
                # Store conversation if session exists (with full RAG logging)
                if conversation_id:
                    logger.info("Storing conversation with RAG logging in conversation ::: {conversation_id}")
                    processing_time_ms = int((time.time() - start_time) * 1000)

                    # Prepare provider info safely
                    provider_name = request.provider if request.use_llm else None
                    model_name = None
                    if request.use_llm and request.provider_specific:
                        model_name = request.provider_specific.get("model_name")

                    await self._store_conversation(
                        session_id=session_id,
                        question=request.question,
                        answer=answer,
                        conversation_id=conversation_id,
                        # RAG Pipeline Data
                        documents=documents,
                        context=context,
                        final_prompt=final_prompt,
                        provider_name=provider_name,
                        model_name=model_name,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        top_k=request.top_k,
                        use_documents=request.use_documents,
                        use_llm=request.use_llm,
                        processing_time_ms=processing_time_ms,
                        error_message=None
                    )

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

    async def retrieve_documents(self, query: str, user: Dict[str, Any], top_k: int = 5,
                                 category: Optional[str] = None) -> List[Dict[str, Any]]:
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

    async def generate_response(self, prompt: str, provider, max_tokens: int = 256,
                                temperature: float = 0.1) -> LLMResponse | None:  # -> Optional[str]:
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

    async def _handle_guest_onboarding(self, question: str, session_id: Optional[str],
                                       conversation_id: Optional[str] = None, user_id: Optional[str] = None) -> \
            Optional[RAGResponse]:
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

        # Get recent messages (try conversation first)
        session_history = []
        if self.conversation_manager and conversation_id and user_id:
            try:
                session_history = await self.conversation_manager.get_messages(conversation_id, user_id, limit=3)
            except Exception as e:
                logger.warning(f"Failed to get history from conversation manager: {e}")
                session_history = self.session_manager.fetch_recent_messages(session_id, limit=3)
        else:
            session_history = self.session_manager.fetch_recent_messages(session_id, limit=3)

        logger.info("Session history: %s", session_history)

        # Check if this is a response to onboarding question
        if session_history:
            last_msg = session_history[-1]
            if (last_msg.get("speaker") == "assistant" and
                    last_msg.get("content", "").strip() == next_field["question"].strip()):

                # Save user response
                if self.conversation_manager and conversation_id:
                    await self.conversation_manager.add_rag_message(
                        conversation_id=conversation_id, speaker="user", content=question,
                        user_query=question, use_llm=False, use_documents=False
                    )

                self.session_manager.set_profile_value(session_id, next_field["key"], question.strip())

                # Check for next question
                next_field = self.session_manager.get_next_missing_profile_key(session_id)
                if next_field:
                    if self.conversation_manager and conversation_id:
                        await self.conversation_manager.add_rag_message(
                            conversation_id=conversation_id, speaker="assistant", content=next_field["question"],
                            llm_response_raw=next_field["question"], use_llm=False, use_documents=False
                        )
                    return RAGResponse(answer=next_field["question"], retrieved_documents=[], context="")
                else:
                    completion_msg = "Thank you! Your details have been saved."
                    if self.conversation_manager and conversation_id:
                        await self.conversation_manager.add_rag_message(
                            conversation_id=conversation_id, speaker="assistant", content=completion_msg,
                            llm_response_raw=completion_msg, use_llm=False, use_documents=False
                        )
                    return RAGResponse(answer=completion_msg, retrieved_documents=[], context="")

        # Ask first onboarding question
        if self.conversation_manager and conversation_id:
            await self.conversation_manager.add_rag_message(
                conversation_id=conversation_id, speaker="assistant", content=next_field["question"],
                llm_response_raw=next_field["question"], use_llm=False, use_documents=False
            )
        return RAGResponse(answer=next_field["question"], retrieved_documents=[], context="")

    async def _get_guest_profile(self, session_id: Optional[str]) -> Dict[str, Any]:
        """Get guest profile from session."""
        if not session_id or not hasattr(self.session_manager, 'get_full_profile'):
            return {}
        return self.session_manager.get_full_profile(session_id) or {}

    async def _get_session_history(self, session_id: Optional[str], user_id: Optional[str] = None,
                                   conversation_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get conversation history from conversation manager or fallback to session."""
        # Try conversation manager first (new approach)
        if self.conversation_manager and conversation_id and user_id:
            try:
                messages = await self.conversation_manager.get_messages(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    limit=5
                )
                logger.debug(f"Retrieved {len(messages)} messages from conversation {conversation_id}")
                return messages
            except Exception as e:
                logger.warning(f"Failed to get conversation history: {e}, falling back to session")

        # Fallback to session-based history (legacy)
        if not session_id or not hasattr(self.session_manager, 'fetch_recent_messages'):
            return []
        return self.session_manager.fetch_recent_messages(session_id, limit=5) or []

    async def _store_conversation(
            self,
            session_id: str,
            question: str,
            answer: str,
            conversation_id: Optional[str] = None,
            # RAG Pipeline Data
            documents: Optional[List[Dict[str, Any]]] = None,
            context: Optional[str] = None,
            final_prompt: Optional[str] = None,
            provider_name: Optional[str] = None,
            model_name: Optional[str] = None,
            temperature: Optional[float] = None,
            max_tokens: Optional[int] = None,
            top_k: Optional[int] = None,
            use_documents: bool = True,
            use_llm: bool = True,
            processing_time_ms: Optional[int] = None,
            error_message: Optional[str] = None
    ):
        """Store conversation with comprehensive RAG logging in conversation manager."""
        # Try conversation manager first (new approach with RAG logging)
        if self.conversation_manager and conversation_id:
            # Store user message (simple)
            await self.conversation_manager.add_message(
                conversation_id=conversation_id,
                speaker="user",
                content=question
            )

            # Store assistant message with full RAG logging
            if answer:
                # Prepare retrieved context for logging
                retrieved_context = None
                retrieved_doc_ids = None
                if documents:
                    retrieved_context = [
                        {
                            "id": doc.get("id", "unknown"),
                            "text": doc.get("text", "")[:500],  # Truncate for storage
                            "metadata": doc.get("metadata", {}),
                            "distance": doc.get("distance")
                        }
                        for doc in documents[:10]  # Limit to top 10
                    ]
                    retrieved_doc_ids = [doc.get("id", "unknown") for doc in documents]

                # Get embedding info (if available)
                embeddings_used = None
                if hasattr(self.vector_store, 'embedding_model'):
                    embeddings_used = {
                        "model": getattr(self.vector_store, 'embedding_model', 'unknown'),
                        "dimensions": getattr(self.vector_store, 'embedding_dimensions', None)
                    }

                await self.conversation_manager.add_rag_message(
                    conversation_id=conversation_id,
                    speaker="assistant",
                    content=answer,
                    # RAG Pipeline Data
                    user_query=question,
                    retrieved_context=retrieved_context,
                    embeddings_used=embeddings_used,
                    llm_prompt=final_prompt,
                    llm_response_raw=answer,
                    llm_provider=provider_name,
                    llm_model=model_name,
                    llm_tokens_used=None,  # TODO: Extract from provider if available
                    llm_temperature=temperature,
                    llm_max_tokens=max_tokens,
                    retrieved_doc_ids=retrieved_doc_ids,
                    retrieval_top_k=top_k,
                    use_documents=use_documents,
                    use_llm=use_llm,
                    processing_time_ms=processing_time_ms,
                    error_message=error_message
                )
            logger.debug(f"Stored conversation with RAG logging in conversation {conversation_id}")

    async def build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context from retrieved documents."""
        if not documents:
            return ""

        context_parts = []
        for i, doc in enumerate(documents[:5]):  # Limit to top 5
            text = doc.get("text", "")
            if text:
                context_parts.append(f"Document {i + 1}: {text[:500]}...")  # Truncate

        return "\n\n".join(context_parts)
