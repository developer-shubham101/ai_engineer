"RAG orchestrator implementation."

import json
import logging
import re
import time
import traceback
from datetime import datetime
from pathlib import Path
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

RAG_DEBUG_LOG_DIR = Path("logs") / "rag_debug"


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

    def _get_conversation_debug_log_path(self, conversation_id: Optional[str]) -> Path:
        """Return the per-conversation RAG debug log path."""
        raw_id = conversation_id or "no_conversation_id"
        safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(raw_id)).strip("._")
        safe_id = safe_id[:120] or "no_conversation_id"
        return RAG_DEBUG_LOG_DIR / f"{safe_id}.rag_debug.log"

    def _json_safe(self, value: Any) -> Any:
        """Convert common Python objects into JSON-serializable debug data."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(v) for v in value]
        if hasattr(value, "__dict__"):
            return self._json_safe(vars(value))
        return repr(value)

    def _request_debug_payload(self, request: RAGRequest) -> Dict[str, Any]:
        """Capture request settings relevant to RAG debugging."""
        return {
            "question": request.question,
            "user": request.user,
            "session_id": request.session_id,
            "conversation_id": request.conversation_id,
            "top_k": request.top_k,
            "use_llm": request.use_llm,
            "use_documents": request.use_documents,
            "use_conversation_history": request.use_conversation_history,
            "enable_agentic_mode": request.enable_agentic_mode,
            "use_tools": request.use_tools,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "category": request.category,
            "debug": request.debug,
            "provider": request.provider,
            "provider_specific": request.provider_specific,
            "metadata": request.metadata,
            "prompt_template": request.prompt_template,
        }

    def _provider_debug_payload(self, provider: Any) -> Dict[str, Any]:
        """Capture provider metadata without letting debug logging affect the RAG flow."""
        payload = {
            "class": provider.__class__.__name__ if provider else None,
            "module": provider.__class__.__module__ if provider else None,
        }
        if not provider:
            return payload

        provider_methods = {
            "provider_name": "get_provider_name",
            "model_name": "get_model_name",
            "is_available": "is_available",
            "max_context_length": "get_max_context_length",
        }
        for field_name, method_name in provider_methods.items():
            try:
                method = getattr(provider, method_name, None)
                payload[field_name] = method() if callable(method) else None
            except Exception as e:
                payload[field_name] = f"debug_read_failed: {e}"
        return payload

    def _write_conversation_debug_log(
            self,
            conversation_id: Optional[str],
            event: str,
            payload: Dict[str, Any],
    ) -> None:
        """Append detailed RAG debug data to a log file scoped by conversation_id."""
        try:
            RAG_DEBUG_LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_path = self._get_conversation_debug_log_path(conversation_id)
            entry = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "event": event,
                "conversation_id": conversation_id,
                "payload": self._json_safe(payload),
            }
            with log_path.open("a", encoding="utf-8") as debug_file:
                debug_file.write(json.dumps(entry, ensure_ascii=False, indent=2))
                debug_file.write("\n\n")
            logger.debug("[RAG DEBUG] Wrote conversation debug event '%s' to %s", event, log_path)
        except Exception as log_error:
            logger.warning(
                "Failed to write conversation RAG debug log for conversation_id=%s event=%s: %s",
                conversation_id,
                event,
                log_error,
            )

    async def process_query(self, request: RAGRequest) -> RAGResponse:
        """Process RAG query with three flows:
        1. Authenticated Company User (non-Guest): Profile from user_meta, session tracking
        2. Authenticated Guest User: Onboarding questions, session tracking
        3. Unauthenticated User: Basic RAG without session
        """
        global prompt_data
        
        # Step 0: Query Preprocessing (before middleware)
        from app.modules.vector_db.query_preprocessor import QueryPreprocessor
        preprocessor = QueryPreprocessor()
        
        processed_query = await preprocessor.process_query(
            query=request.question,
            use_spell_correction=True,
            use_expansion=False,
            use_llm_rewrite=False
        )
        
        # Log preprocessing results
        logger.info(f"Query preprocessing: original='{processed_query.original}'")
        logger.info(f"Query type: {processed_query.query_type.value}")
        if processed_query.corrected:
            logger.info(f"Spell correction: '{processed_query.corrected}'")
        logger.info(f"Total query variants: {len(processed_query.all_variants)}")
        
        # Update request with corrected query if available
        if processed_query.corrected:
            request.question = processed_query.corrected
        
        request = await self.middleware_stack.process_request(request)

        start_time = time.time()

        if request.metadata and "_cached_response" in request.metadata:
            return await self.middleware_stack.process_response(request, request.metadata["_cached_response"])

        # # Handle tool-based agent mode
        if request.use_tools:
            from app.modules.agents.agent_runner import run_agent

            try:
                # Get LLM provider for agent
                provider_config = request.provider_specific or {}
                llm_provider = await create_provider(request.provider, provider_config)

                # Run agent with tools
                agent_response = await run_agent(request.question, llm_provider)

                return RAGResponse(
                    answer=agent_response,
                    retrieved_documents=[],
                    context="Agent mode with tools",
                    final_prompt=None,
                    metadata={"provider": request.provider, "mode": "agent_tools"}
                )
            except Exception as e:
                logger.exception("Agent tool execution failed: %s", e)
                error_response = RAGResponse(
                    answer=f"Agent error: {str(e)}",
                    retrieved_documents=[],
                    context="",
                    metadata={"error": str(e), "mode": "agent_tools"}
                )
                return await self.middleware_stack.process_response(request, error_response)

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

            self._write_conversation_debug_log(
                conversation_id=conversation_id,
                event="rag_request_start",
                payload={
                    "request": self._request_debug_payload(request),
                    "session_id": session_id,
                    "user_id": user_id,
                    "user_role": user_role,
                    "profile": profile,
                    "session_history_count": len(session_history),
                    "session_history": session_history,
                },
            )

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

            self._write_conversation_debug_log(
                conversation_id=conversation_id,
                event="rag_retrieval_complete",
                payload={
                    "use_documents": request.use_documents,
                    "top_k": request.top_k,
                    "category": request.category,
                    "retrieved_document_count": len(documents),
                    "retrieved_documents": documents,
                    "context_length_chars": len(context or ""),
                    "context": context,
                },
            )

            # Generate response if LLM requested
            answer = None
            final_prompt = None
            if request.use_llm:
                logger.info("Generating response using LLM provider: %s", request.provider)
                provider_config = request.provider_specific or {}
                provider = await create_provider(request.provider, provider_config)

                # Get user context
                user_role = request.user.get("role", "Guest") if request.user else "Guest"
                department = request.user.get("department", "General") if request.user else "General"

                # Build messages array directly
                messages = await self._build_messages(
                    template_name=request.prompt_template,
                    user_question=request.question,
                    documents=documents if request.use_documents else [],
                    history=session_history if request.use_conversation_history else [],
                    user_role=user_role,
                    department=department,
                    user_profile=profile,
                    max_tokens=request.max_tokens
                )
                
                logger.debug(f"[RAG DEBUG] Generated {messages} messages for LLM")
                self._write_conversation_debug_log(
                    conversation_id=conversation_id,
                    event="llm_messages_built",
                    payload={
                        "provider": request.provider,
                        "provider_details": self._provider_debug_payload(provider),
                        "provider_specific": request.provider_specific or {},
                        "prompt_template": request.prompt_template,
                        "max_tokens": request.max_tokens,
                        "temperature": request.temperature,
                        "message_count": len(messages),
                        "message_roles": [msg.get("role") for msg in messages],
                        "messages": messages,
                        "formatted_prompt": self._format_messages_for_debug(messages),
                    },
                )
                
                # Generate response using messages directly
                response = await provider.generate(
                    prompt=messages,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature
                )

                self._write_conversation_debug_log(
                    conversation_id=conversation_id,
                    event="llm_provider_response",
                    payload={
                        "provider": request.provider,
                        "provider_details": self._provider_debug_payload(provider),
                        "response": {
                            "text": getattr(response, "text", None),
                            "text_length_chars": len(getattr(response, "text", "") or ""),
                            "metadata": getattr(response, "metadata", None),
                            "usage": getattr(response, "usage", None),
                            "finish_reason": getattr(response, "finish_reason", None),
                        } if response else None,
                    },
                )
                
                if response and response.text:
                    answer = response.text
                    final_prompt = self._format_messages_for_debug(messages)
                    logger.debug(f"[RAG DEBUG] LLM response length: {len(answer)} characters")
                else:
                    answer = "I found relevant documents but couldn't generate a response"
                    final_prompt = self._format_messages_for_debug(messages)
                    logger.warning("[RAG DEBUG] LLM returned empty response")

                log_timestamp = datetime.now().isoformat()
                logger.info(f"""
                [DEBUG LOG - {log_timestamp}]
                ==================================================
                Final AI Response: {answer}
                ==================================================
                """)

                logger.info("Storing conversation with RAG logging in conversation %s", conversation_id)
                # Store conversation if session exists (with full RAG logging)
                if conversation_id:
                    logger.info("Storing conversation with RAG logging in conversation: %s", conversation_id)
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

            self._write_conversation_debug_log(
                conversation_id=conversation_id,
                event="rag_response_complete",
                payload={
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                    "answer_length_chars": len(answer or ""),
                    "answer": answer,
                    "retrieved_document_count": len(retrieved_docs),
                    "final_prompt": final_prompt,
                    "response_metadata": response.metadata,
                },
            )

            return await self.middleware_stack.process_response(request, response)

        except Exception as e:
            logger.error("RAG processing failed: %s", str(e), exc_info=True)
            self._write_conversation_debug_log(
                conversation_id=getattr(request, "conversation_id", None),
                event="rag_error",
                payload={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "traceback": traceback.format_exc(),
                    "request": self._request_debug_payload(request),
                },
            )
            error_response = RAGResponse(
                answer="Sorry, I encountered an error processing your request.",
                retrieved_documents=[],
                context="",
                metadata={"error": str(e)}
            )
            return await self.middleware_stack.process_response(request, error_response)

    async def retrieve_documents(self, query: str, user: Dict[str, Any], top_k: int = 5,
                                 category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents with hybrid BM25+vector search and cross-encoder reranking.
        
        Pipeline:
        1. Retrieve top-20 from both BM25 and vector store
        2. Merge with Reciprocal Rank Fusion (RRF)
        3. Apply RBAC filtering
        4. Rerank with cross-encoder
        5. Return top-k documents
        """
        try:
            # Step 1: Hybrid retrieval (BM25 + Vector)
            retrieval_k = max(top_k * 4, 20)  # Retrieve 4x or minimum 20
            logger.info(f"Hybrid retrieval: fetching top-{retrieval_k} from BM25 and vector store")
            
            # Get BM25 index from container
            from app.modules.integration import get_container
            container = get_container()
            bm25_index = container.get_bm25_index()
            
            # BM25 search
            bm25_results = []
            if bm25_index and bm25_index.is_available():
                bm25_results = bm25_index.search(query, top_k=retrieval_k)
                logger.info(f"BM25 retrieved {len(bm25_results)} documents")
            
            # Vector search
            vector_results = await self.vector_store.search_documents(
                query=query,
                top_k=retrieval_k,
                metadata_filter={"category": category} if category else None
            )
            logger.info(f"Vector search retrieved {len(vector_results)} documents")
            
            # Step 2: Merge with RRF
            from app.modules.vector_db.hybrid_retrieval import reciprocal_rank_fusion
            merged_results = reciprocal_rank_fusion(bm25_results, vector_results, k=60)
            logger.info(f"RRF fusion: {len(bm25_results)} BM25 + {len(vector_results)} vector → {len(merged_results)} merged")

            # Step 3: Apply RBAC filtering
            filtered_results = []
            user_role = user.get("role", "Guest")
            user_dept = user.get("department", "General")

            for doc in merged_results:
                metadata = doc.get("metadata", {})
                doc_dept = metadata.get("department", "General")
                sensitivity = metadata.get("sensitivity", "public_internal")

                if sensitivity == "public_internal" or doc_dept == user_dept or user_role in ["SuperAdmin", "Manager"]:
                    filtered_results.append(doc)
            
            logger.info(f"RBAC filtering: {len(merged_results)} -> {len(filtered_results)} documents")

            # Step 4: Rerank with cross-encoder
            if len(filtered_results) > top_k:
                try:
                    from app.modules.vector_db.reranker import CrossEncoderReranker
                    reranker = CrossEncoderReranker()
                    reranked_results = reranker.rerank(
                        query=query,
                        documents=filtered_results,
                        top_k=top_k
                    )
                    logger.info(f"Cross-encoder reranking: {len(filtered_results)} -> {top_k} documents")
                    return reranked_results
                except Exception as e:
                    logger.warning(f"Reranking failed, using RRF scores: {e}")
                    return filtered_results[:top_k]
            else:
                return filtered_results[:top_k]

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
                    limit=55
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

    async def _build_messages(self, template_name: str, user_question: str, 
                             documents: List[Dict[str, Any]], history: List[Dict[str, Any]],
                             user_role: str, department: str, user_profile: Dict[str, Any],
                             max_tokens: int) -> List[Dict[str, str]]:
        """Build messages: system (template 1st), history (optional), user (template 2nd)."""
        logger.info(f"[MESSAGE BUILD] ========== Starting Message Build ==========")
        logger.info(f"[MESSAGE BUILD] Template: '{template_name}'")
        logger.info(f"[MESSAGE BUILD] User Question: '{user_question[:100]}...'")
        logger.info(f"[MESSAGE BUILD] Documents count: {len(documents)}")
        logger.info(f"[MESSAGE BUILD] History count: {len(history)}")
        
        messages = []
        
        # Prepare variables
        source_docs = "\n\n".join([
            f"Document {i+1}: {doc.get('text', '')[:500]}..."
            for i, doc in enumerate(documents[:5])
        ]) if documents else ""
        
        variables = {
            'user_question': user_question,
            'source_docs': source_docs,
            'user_role': user_role,
            'department': department,
            'user_profile_summary': str(user_profile) if user_profile else "No profile",
            'max_tokens': str(max_tokens)
        }
        logger.info(f"[MESSAGE BUILD] Variables: {list(variables.keys())}")
        
        # Get template
        template_obj = None
        if template_name and not template_name.startswith("SYSTEM:"):
            from ..integration import get_container
            container = get_container()
            template_manager = container.get_template_manager()
            template_obj = template_manager.get_template(template_name)
            logger.info(f"[MESSAGE BUILD] Template found: {template_obj is not None}")
        
        # 1. System message (first element from template)
        if template_obj and template_obj.get('messages') and len(template_obj['messages']) > 0:
            system_msg = template_obj['messages'][0].copy()
            content = system_msg['content']
            for var_name, var_value in variables.items():
                content = content.replace(f'{{{var_name}}}', var_value)
            system_msg['content'] = content
            messages.append(system_msg)
            logger.info(f"[MESSAGE BUILD] Added system message from template")
        else:
            messages.append({
                "role": "system",
                "content": f"You are a helpful enterprise assistant. User role: {user_role}, Department: {department}"
            })
            logger.info(f"[MESSAGE BUILD] Added default system message")
        
        # 2. History (optional)
        if history:
            for msg in history:
                role = "user" if msg.get("speaker") == "user" else "assistant"
                messages.append({"role": role, "content": msg.get("content", "")})
            logger.info(f"[MESSAGE BUILD] Added {len(history)} history messages")
        
        # 3. User message (second element from template)
        if template_obj and template_obj.get('messages') and len(template_obj['messages']) > 1:
            user_msg = template_obj['messages'][1].copy()
            content = user_msg['content']
            for var_name, var_value in variables.items():
                content = content.replace(f'{{{var_name}}}', var_value)
            user_msg['content'] = content
            messages.append(user_msg)
            logger.info(f"[MESSAGE BUILD] Added user message from template")
        else:
            messages.append({"role": "user", "content": user_question})
            logger.info(f"[MESSAGE BUILD] Added default user message")
        
        logger.info(f"[MESSAGE BUILD] ========== Complete ==========")
        logger.info(f"[MESSAGE BUILD] Final: {messages}")
        logger.info(f"[MESSAGE BUILD] Roles: {[msg.get('role') for msg in messages]}")
        
        return messages

    
    def _format_messages_for_debug(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for debug logging."""
        return "\n\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in messages])
    
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
