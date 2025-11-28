# app/services/base_rag_service.py
"""
Base RAG service containing common functionality for all RAG providers.
This service handles document retrieval, RBAC filtering, session management,
and response formatting that can be shared across local, Google, GPT, and other providers.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

from app.logging_config import log_sensitive_debug
from app.services.chroma_utils import ensure_chroma_client, query_collection
from app.services.support_chat import fetch_recent_messages
from app.services.prompt_builder import build_tone_guidance, estimate_tokens_from_text
from app.services.utility import embed_texts, DEFAULT_PERSIST_DIR, DEFAULT_COLLECTION_NAME
# Removed old profile_analyzer import - using optimized approach

logger = logging.getLogger(__name__)


class BaseRAGService(ABC):
    """
    Abstract base class for RAG services.
    Contains common functionality that all RAG providers can use.
    """
    
    def __init__(self):
        self.persist_directory = str(DEFAULT_PERSIST_DIR)
        self.collection_name = DEFAULT_COLLECTION_NAME
    
    async def retrieve_documents(self, query_text: str, n_results: int) -> Dict[str, Any]:
        """
        Retrieve documents from ChromaDB based on query embedding.
        Common across all RAG providers.
        """
        client, collection = ensure_chroma_client(
            persist_directory=self.persist_directory,
            collection_name=self.collection_name
        )

        try:
            q_emb = (await embed_texts([query_text]))[0]
            logger.debug("Computed query embedding.")
        except Exception as e:
            logger.exception("Failed to embed query: %s", e)
            raise

        try:
            result = query_collection(collection=collection, query_embeddings=[q_emb], n_results=n_results)
        except Exception:
            result = query_collection(collection=collection, query_texts=[query_text], n_results=n_results)

        return result
    
    def normalize_chroma_result(self, result: Any) -> Tuple[List[str], List[Dict], List[str], List[float]]:
        """
        Normalize ChromaDB result into standard lists.
        Common across all RAG providers.
        """
        if isinstance(result, dict):
            raw_docs = (result.get("documents") or [[]])[0]
            raw_metadatas = (result.get("metadatas") or [[]])[0]
            raw_ids = (result.get("ids") or [[]])[0]
            raw_distances = (result.get("distances") or [[]])[0]
        else:
            try:
                raw_docs = result.documents[0]
                raw_metadatas = result.metadatas[0]
                raw_ids = result.ids[0]
                raw_distances = result.distances[0] if hasattr(result, "distances") else []
            except Exception as e:
                logger.exception("Unexpected Chroma format: %s", e)
                raw_docs, raw_metadatas, raw_ids, raw_distances = [], [], [], []
        
        return raw_docs, raw_metadatas, raw_ids, raw_distances
    
    def _allowed_by_metadata(self, meta: Optional[Dict[str, Any]], requester: Optional[Dict[str, str]]) -> bool:
        """
        Flexible RBAC: Level-based access + specific role overrides.
        """
        if not requester:
            logger.debug("RBAC_CHECK: No requester provided. Defaulting to public_internal.")
            return meta.get("sensitivity", "public_internal") == "public_internal"
        
        sens = meta.get("sensitivity", "public_internal") if meta else "public_internal"
        user_role = requester.get("role")
        user_level = self._get_role_level(user_role)

        logger.debug("RBAC_CHECK: user_role=%s user_level=%d doc_sensitivity=%s", user_role, user_level, sens)
        
        # Personal documents: owner or high-level roles
        if sens == "personal":
            owner = meta.get("owner_id")
            log_sensitive_debug(logger, "RBAC_CHECK: personal doc owner=", owner=owner)
            if owner == requester.get("user_id"):
                log_sensitive_debug(logger, "RBAC_CHECK: personal doc owner match")
                return True
            log_sensitive_debug(logger, "RBAC_CHECK: personal doc owner mismatch")
            return user_level >= 2  # HR+
        
        # Specific role restrictions override level hierarchy
        allowed_roles = meta.get("allowed_roles")
        if allowed_roles:
            log_sensitive_debug(logger, "RBAC_CHECK: allowed_roles=", allowed_roles =allowed_roles)
            # SuperAdmin bypasses role restrictions
            if user_role == "SuperAdmin":
                return True
            return user_role in allowed_roles
        
        # Department restrictions
        if sens == "department_confidential":
            log_sensitive_debug(logger, "RBAC_CHECK: department_confidential doc dept", department= meta.get("department"))
            if requester.get("department") == meta.get("department"):
                log_sensitive_debug(logger, "RBAC_CHECK: department_confidential doc dept match")
                return True
        
        # Default: Level-based access
        required_level = self._get_sensitivity_level(sens)
        has_access = user_level >= required_level
        logger.debug("RBAC_RESULT: user_level=%d required_level=%d access=%s", user_level, required_level, has_access)
        return has_access
    
    def _get_role_level(self, role: str) -> int:
        """Get numerical level for role."""
        levels = {"SuperAdmin": 4, "Manager": 3, "HR": 2, "Employee": 1, "PublicUser": 0, "Guest": 0}
        return levels.get(role, 0)
    
    def _get_sensitivity_level(self, sensitivity: str) -> int:
        """Get required level for sensitivity."""
        levels = {"public_internal": 0, "department_confidential": 1, "role_confidential": 2, "highly_confidential": 3, "super_confidential": 4}
        return levels.get(sensitivity, 0)
    
    def filter_documents_by_rbac(
        self,
        raw_docs: List[str],
        raw_metadatas: List[Dict],
        raw_ids: List[str],
        raw_distances: List[float],
        requester: Optional[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Filter documents based on RBAC rules AND deduplicate versions (show only latest accessible).
        Common across all RAG providers.
        """
        temp_docs, temp_metas, temp_ids, temp_distances = [], [], [], []
        public_summaries, filtered_details = [], []
        filtered_out_count = 0

        logger.debug("Requester for RBAC filtering: %s", requester or "anonymous")
        #{'user_id': 'u_admin_1', 'username': 'admin', 'role': 'SuperAdmin', 'department': 'Executive', 'session_id': 'sess_0bc192002dd44bc78ff272f1c534cb03', 'exp': 1764413396, 'iat': 1764326996}


        # 1. First Pass: RBAC Filtering
        for doc, meta, id_, dist in zip(raw_docs, raw_metadatas, raw_ids, raw_distances):
            try:
                has_access = self._allowed_by_metadata(meta, requester)
                logger.debug("RBAC check for document_id=%s access=%s", id_, has_access)
                if has_access:
                    log_sensitive_debug(
                        logger, "RBAC allowed document",
                        document_id=id_, document=doc, metadata=meta
                    )
                    temp_docs.append(doc)
                    temp_metas.append(meta)
                    temp_ids.append(id_)
                    temp_distances.append(dist)
                else:
                    log_sensitive_debug(
                        logger, "RBAC blocked document",
                        document_id=id_, document=doc, metadata=meta
                    )
                    filtered_out_count += 1
                    ps = meta.get("public_summary") if isinstance(meta, dict) else None
                    if ps:
                        public_summaries.append(ps)
                    filtered_details.append({
                        "id": id_,
                        "sensitivity": meta.get("sensitivity"),
                        "department": meta.get("department"),
                        "source": meta.get("source"),
                    })
            except Exception as e:
                logger.exception("Metadata filtering error: %s", e)

        # Audit logging for blocked access attempts
        if filtered_out_count > 0:
            logger.debug("RBAC filtering blocked %d documents", filtered_out_count)
            from app.logging_config import log_security_event
            
            user_id = requester.get("user_id", "anonymous") if requester else "anonymous"
            user_role = requester.get("role", "none") if requester else "none"
            user_dept = requester.get("department", "none") if requester else "none"
            
            blocked_sources = [d.get("source", "unknown")[:50] for d in filtered_details[:3]]
            blocked_sensitivities = [d.get("sensitivity", "unknown") for d in filtered_details[:3]]
            
            log_security_event(
                logger, "RBAC_ACCESS_DENIED", user_id,
                role=user_role, department=user_dept, filtered_count=filtered_out_count,
                blocked_sources=blocked_sources, blocked_sensitivities=blocked_sensitivities,
                provider=self.__class__.__name__
            )

        # 2. Second Pass: Version Deduplication (Show only latest accessible version)
        doc_groups = {}
        
        for i, meta in enumerate(temp_metas):
            doc_id = meta.get("document_id")
            if not doc_id:
                continue
                
            version_str = meta.get("version", "1.0")
            try:
                version_val = float(version_str)
            except ValueError:
                version_val = 1.0
                
            if doc_id not in doc_groups:
                doc_groups[doc_id] = {"max_ver": version_val, "indices": [i]}
            else:
                current_max = doc_groups[doc_id]["max_ver"]
                if version_val > current_max:
                    doc_groups[doc_id]["max_ver"] = version_val
                elif version_val < current_max:
                    pass # Don't update max
                
                doc_groups[doc_id]["indices"].append(i)

        # Build final list
        visible_docs, visible_metas, visible_ids, visible_distances = [], [], [], []
        
        for i in range(len(temp_docs)):
            meta = temp_metas[i]
            doc_id = meta.get("document_id")
            if not doc_id:
                # No ID, keep it (safe fallback)
                visible_docs.append(temp_docs[i])
                visible_metas.append(temp_metas[i])
                visible_ids.append(temp_ids[i])
                visible_distances.append(temp_distances[i])
                continue
                
            version_str = meta.get("version", "1.0")
            try:
                version_val = float(version_str)
            except ValueError:
                version_val = 1.0
                
            max_ver = doc_groups[doc_id]["max_ver"]
            
            if version_val >= max_ver:
                visible_docs.append(temp_docs[i])
                visible_metas.append(temp_metas[i])
                visible_ids.append(temp_ids[i])
                visible_distances.append(temp_distances[i])

        return {
            "documents": visible_docs,
            "metadatas": visible_metas,
            "ids": visible_ids,
            "distances": visible_distances,
            "filtered_out_count": filtered_out_count,
            "public_summaries": public_summaries,
            "filtered_details": filtered_details
        }
    
    def inject_personalized_context(
        self, 
        session_id: Optional[str], 
        llm_prompt_prefix: Optional[str],
        query_text: str,
        requester: Optional[Dict[str, str]],
        profile: Optional[Dict[str, str]] = None,
        max_prefix_tokens: int = 80  # Token budget for system prefix
    ) -> str:
        """
        Build highly optimized prompt with strict token budgeting.
        Prioritizes essential information and maintains conversation context.
        """
        from app.logging_config import log_performance_metric
        import time
        start_time = time.time()
        
        logger.info("PROMPT_OPTIMIZATION_START: session_id=%s query_len=%d max_prefix_tokens=%d", 
                   session_id, len(query_text or ""), max_prefix_tokens)
        
        # Get user context
        user_role = (requester or {}).get("role", "Guest")
        user_dept = (requester or {}).get("department", "General")
        logger.debug("USER_CONTEXT: role=%s dept=%s session=%s", user_role, user_dept, session_id or "none")
        
        # Get conversation context and tone
        chat_history = []
        last_user_tone = None
        
        if session_id:
            try:
                chat_history = fetch_recent_messages(session_id, limit=2)  # Reduced for efficiency
                logger.debug("CHAT_HISTORY: fetched %d messages for session=%s", len(chat_history), session_id)
                # Find last user tone
                for m in reversed(chat_history):
                    if m.get("speaker") == "user" and m.get("tone"):
                        last_user_tone = m["tone"]
                        logger.debug("DETECTED_TONE: %s for session=%s", last_user_tone, session_id)
                        break
            except Exception as e:
                logger.warning("History fetch failed: %s", e)

        # Build ultra-compact prompt with token budgeting
        prompt_parts = []
        current_tokens = 0
        
        # 1. Essential system instruction (compressed)
        system_base = "Assistant for Saarthi Infotech"
        system_tokens = estimate_tokens_from_text(system_base)
        
        if current_tokens + system_tokens <= max_prefix_tokens:
            prompt_parts.append(system_base)
            current_tokens += system_tokens
            logger.debug("ADDED_SYSTEM: %s (%d tokens)", system_base, system_tokens)
        
        # 2. User context (ultra-compact)
        if user_role != "Guest" or user_dept != "General":
            user_context = f"{user_role}/{user_dept}"
            user_tokens = estimate_tokens_from_text(user_context)
            
            if current_tokens + user_tokens <= max_prefix_tokens:
                prompt_parts.append(user_context)
                current_tokens += user_tokens
                logger.debug("ADDED_USER_CONTEXT: %s (%d tokens)", user_context, user_tokens)
        
        # 3. Critical profile info only (name/position if space allows)
        if profile and current_tokens < max_prefix_tokens - 10:  # Reserve 10 tokens
            profile_items = []
            
            # Prioritize name and position only
            if profile.get('name'):
                profile_items.append(profile['name'])
            if profile.get('position') and len(profile_items) == 0:  # Only if no name
                profile_items.append(profile['position'])
            
            if profile_items:
                profile_text = profile_items[0]  # Take only the first item
                profile_tokens = estimate_tokens_from_text(profile_text)
                
                if current_tokens + profile_tokens <= max_prefix_tokens:
                    prompt_parts.append(profile_text)
                    current_tokens += profile_tokens
                    logger.debug("ADDED_PROFILE: %s (%d tokens)", profile_text, profile_tokens)
        
        # 4. Tone guidance (compressed)
        if last_user_tone and current_tokens < max_prefix_tokens - 15:
            tone_guidance = build_tone_guidance(last_user_tone)
            # Compress tone guidance to essential words only
            tone_compressed = tone_guidance.split('.')[0]  # Take first sentence only
            tone_tokens = estimate_tokens_from_text(tone_compressed)
            
            if current_tokens + tone_tokens <= max_prefix_tokens:
                prompt_parts.append(tone_compressed)
                current_tokens += tone_tokens
                logger.debug("ADDED_TONE: %s (%d tokens)", tone_compressed, tone_tokens)
        
        # 5. Last exchange context (only if significant space remains)
        if chat_history and current_tokens < max_prefix_tokens - 20:
            if len(chat_history) >= 2:
                recent_msgs = chat_history[-2:]
                user_msg = next((m for m in recent_msgs if m.get("speaker") == "user"), None)
                
                if user_msg:
                    # Ultra-compressed context
                    prev_content = user_msg.get('content', '')[:30]  # Very short
                    if prev_content and prev_content != query_text[:30]:  # Avoid duplication
                        context_text = f"Prev: {prev_content}"
                        context_tokens = estimate_tokens_from_text(context_text)
                        
                        if current_tokens + context_tokens <= max_prefix_tokens:
                            prompt_parts.append(context_text)
                            current_tokens += context_tokens
                            logger.debug("ADDED_CONTEXT: %s (%d tokens)", context_text, context_tokens)
        
        # Build final optimized prefix
        if prompt_parts:
            optimized_prefix = " | ".join(prompt_parts)  # Use compact separator
        else:
            optimized_prefix = "Assistant"  # Fallback minimal prefix
        
        # Calculate final metrics
        final_tokens = estimate_tokens_from_text(optimized_prefix)
        efficiency = (final_tokens / max_prefix_tokens) * 100
        
        optimization_time = (time.time() - start_time) * 1000
        
        log_performance_metric(
            logger, "PROMPT_OPTIMIZATION", optimization_time,
            prefix_len=len(optimized_prefix), tokens_used=final_tokens, 
            max_tokens=max_prefix_tokens, efficiency_pct=efficiency, 
            parts_count=len(prompt_parts), session_id=session_id
        )
        
        logger.info("PROMPT_OPTIMIZATION_COMPLETE: prefix_len=%d tokens=%d/%d (%.1f%%) parts=%d", 
                   len(optimized_prefix), final_tokens, max_prefix_tokens, efficiency, len(prompt_parts))
        
        # Warning if still too large
        if final_tokens > max_prefix_tokens:
            logger.warning("PREFIX_BUDGET_EXCEEDED: %d tokens > %d limit (session=%s)", 
                         final_tokens, max_prefix_tokens, session_id or "none")
        
        log_sensitive_debug(
            logger, "Final optimized prompt prefix", 
            optimized_prefix=optimized_prefix, session_id=session_id
        )
        return optimized_prefix
    
    def build_context_text(self, visible_docs: List[str]) -> str:
        """
        Build context text from visible documents.
        Common across all RAG providers.
        """
        return "\n\n---\n\n".join(visible_docs or [])
    
    def handle_rbac_blocked_response(self, filtered_out_count: int, public_summaries: List[str]) -> str:
        """
        Generate appropriate response when all documents are blocked by RBAC.
        Common across all RAG providers.
        """
        if filtered_out_count > 0:
            if public_summaries:
                # Show public summaries as fallback
                return (
                    "I found relevant information, but you don't have access to the full details. "
                    "Here's what I can share:\n\n" + 
                    "\n".join(f"• {s}" for s in public_summaries)
                )
            else:
                # No public summaries available
                return (
                    "You do not have permission to view this information. "
                    "Please contact your administrator if you believe this is an error."
                )
        return "No relevant documents found in the knowledge base."
    
    def build_base_response(
        self,
        visible_docs: List[str],
        filtered_result: Dict[str, Any],
        raw_docs: List[str],
        raw_metadatas: List[Dict],
        raw_ids: List[str],
        raw_distances: List[float],
        context_text: str,
        answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Build the base response structure common to all RAG providers.
        """
        return {
            "documents": visible_docs,
            "metadatas": filtered_result["metadatas"],
            "ids": filtered_result["ids"],
            "distances": filtered_result["distances"],
            "raw_documents": raw_docs,
            "raw_metadatas": raw_metadatas,
            "raw_ids": raw_ids,
            "raw_distances": raw_distances,
            "context": context_text,
            "filtered_out_count": filtered_result["filtered_out_count"],
            "public_summaries": filtered_result["public_summaries"],
            "filtered_details": filtered_result["filtered_details"],
            "answer": answer
        }
    
    @abstractmethod
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
        Generate a response using the provider-specific LLM.
        Must be implemented by each RAG provider.
        """
        pass
    
    async def query_rag(
        self,
        query_text: str,
        n_results: int = 3,
        requester: Optional[Dict[str, str]] = None,
        llm_prompt_prefix: Optional[str] = None,
        use_llm: bool = True,
        max_tokens: int = 256,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main RAG query method that orchestrates the common flow.
        Uses the template method pattern - calls abstract generate_response method.
        """
        from app.logging_config import log_user_action, log_sensitive_debug
        
        user_id = (requester or {}).get("user_id", "anonymous")
        user_role = (requester or {}).get("role", "none")
        user_dept = (requester or {}).get("department", "none")
        
        log_user_action(
            logger, "RAG_QUERY_START", user_id,
            query_len=len(query_text or ""), n_results=n_results, use_llm=use_llm, 
            max_tokens=max_tokens, session_id=session_id, role=user_role, department=user_dept,
            provider=self.__class__.__name__
        )
        
        # Log sensitive debug info (remove in production)
        log_sensitive_debug(
            logger, "RAG query details", 
            query_text=query_text, user_context=requester
        )

        if not query_text:
            raise ValueError("query_text must be provided")

        # 1. Retrieve
        raw_result = await self.retrieve_documents(query_text, n_results)
        raw_docs, raw_metadatas, raw_ids, raw_distances = self.normalize_chroma_result(raw_result)

        # 2. Filter (RBAC)
        filtered_result = self.filter_documents_by_rbac(raw_docs, raw_metadatas, raw_ids, raw_distances, requester)
        
        visible_docs = filtered_result["documents"]
        filtered_out_count = filtered_result.get("filtered_out_count", 0)
        public_summaries = filtered_result.get("public_summaries", [])
        
        # Handle case where all documents are blocked by RBAC
        if not visible_docs and filtered_out_count > 0:
            answer = self.handle_rbac_blocked_response(filtered_out_count, public_summaries)
            return self.build_base_response(
                visible_docs, filtered_result, raw_docs, raw_metadatas, raw_ids, raw_distances, None, answer
            )
        
        # 3. Build context
        context_text = self.build_context_text(visible_docs)

        # 4. Enhanced Personalization (Tone + Profile Analysis)
        # Get user profile if available
        user_profile = None
        if requester and requester.get("user_id"):
            try:
                from app.services.user_service import get_all_user_meta
                user_profile = get_all_user_meta(requester["user_id"])
            except Exception as e:
                logger.warning("Failed to load user profile: %s", e)
        elif session_id:
            try:
                from app.services.support_chat import get_full_profile
                user_profile = get_full_profile(session_id)
            except Exception as e:
                logger.warning("Failed to load session profile: %s", e)
        
        final_prefix = self.inject_personalized_context(
            session_id, llm_prompt_prefix, query_text, requester, user_profile
        )

        # 5. Generate (provider-specific)
        # DEBUG: Log final query components sent to LLM
        context_len = len(context_text or "")
        prefix_len = len(final_prefix)
        query_len = len(query_text)
        
        # Estimate total tokens before LLM call
        total_input_tokens = estimate_tokens_from_text(final_prefix + (context_text or "") + query_text)
        
        from app.logging_config import log_llm_interaction, log_sensitive_debug
        
        user_id = (requester or {}).get("user_id", "anonymous")
        provider_name = self.__class__.__name__
        
        log_llm_interaction(
            logger, provider_name, total_input_tokens, 0,  # response tokens unknown yet
            user_id=user_id, query_len=query_len, context_len=context_len, 
            prefix_len=prefix_len, session_id=session_id
        )
        
        logger.info(
            "LLM_QUERY_DEBUG: user=%s provider=%s query_len=%d context_len=%d prefix_len=%d total_input_tokens_est=%d",
            user_id, provider_name, query_len, context_len, prefix_len, total_input_tokens
        )
        
        # Log components separately for debugging (sensitive data)
        log_sensitive_debug(
            logger, "LLM prompt components",
            final_prefix=final_prefix, context_text=context_text or "[NO_CONTEXT]", 
            query_text=query_text, provider=provider_name
        )
        
        # Log token efficiency metrics
        if context_len > 0:
            context_tokens = estimate_tokens_from_text(context_text)
            prefix_tokens = estimate_tokens_from_text(final_prefix)
            query_tokens = estimate_tokens_from_text(query_text)
            
            logger.info(
                "TOKEN_BREAKDOWN: prefix=%d context=%d query=%d total=%d max_gen=%d efficiency=%.2f%%",
                prefix_tokens, context_tokens, query_tokens, total_input_tokens, max_tokens,
                (context_tokens / max(total_input_tokens, 1)) * 100
            )
        
        answer = await self.generate_response(
            query_text, context_text, final_prefix, use_llm, max_tokens, session_id
        )

        response = self.build_base_response(
            visible_docs, filtered_result, raw_docs, raw_metadatas, raw_ids, raw_distances, context_text, answer
        )
        
        # Add final prompt if available
        final_prompt = getattr(self, '_last_final_prompt', None) if use_llm else None
        response["final_prompt"] = final_prompt
        return response