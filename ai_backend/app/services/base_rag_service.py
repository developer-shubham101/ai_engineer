# app/services/base_rag_service.py
"""
Base RAG service containing common functionality for all RAG providers.
This service handles document retrieval, RBAC filtering, session management,
and response formatting that can be shared across local, Google, GPT, and other providers.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod

from app.services.chroma_utils import ensure_chroma_client, query_collection
from app.services.support_chat import fetch_recent_messages
from app.services.prompt_builder import build_tone_guidance, estimate_tokens_from_text
from app.services.utility import embed_texts, DEFAULT_PERSIST_DIR, DEFAULT_COLLECTION_NAME
from app.services.profile_analyzer import build_personalized_prompt

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
            return meta.get("sensitivity", "public_internal") == "public_internal"
        
        sens = meta.get("sensitivity", "public_internal") if meta else "public_internal"
        user_role = requester.get("role")
        user_level = self._get_role_level(user_role)
        
        # Personal documents: owner or high-level roles
        if sens == "personal":
            owner = meta.get("owner_id")
            if owner == requester.get("user_id"):
                return True
            return user_level >= 2  # HR+
        
        # Specific role restrictions override level hierarchy
        allowed_roles = meta.get("allowed_roles")
        if allowed_roles:
            return user_role in allowed_roles
        
        # Department restrictions
        if sens == "department_confidential":
            if requester.get("department") == meta.get("department"):
                return True
        
        # Default: Level-based access
        required_level = self._get_sensitivity_level(sens)
        return user_level >= required_level
    
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

        # 1. First Pass: RBAC Filtering
        for doc, meta, id_, dist in zip(raw_docs, raw_metadatas, raw_ids, raw_distances):
            try:
                if self._allowed_by_metadata(meta, requester):
                    temp_docs.append(doc)
                    temp_metas.append(meta)
                    temp_ids.append(id_)
                    temp_distances.append(dist)
                else:
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
            logger.warning(
                "RBAC_ACCESS_DENIED: user_id=%s role=%s dept=%s filtered_count=%d blocked_sources=%s",
                requester.get("user_id", "anonymous") if requester else "anonymous",
                requester.get("role", "none") if requester else "none",
                requester.get("department", "none") if requester else "none",
                filtered_out_count,
                [d.get("source", "unknown")[:50] for d in filtered_details[:3]]
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
        profile: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Inject personalized context including tone guidance and profile analysis.
        Enhanced version of inject_tone_guidance with profile support.
        """
        chat_history = []
        last_user_tone = None
        
        if session_id:
            try:
                chat_history = fetch_recent_messages(session_id, limit=10)
                for m in reversed(chat_history):
                    if m.get("speaker") == "user" and m.get("tone"):
                        last_user_tone = m["tone"]
                        break
            except Exception as e:
                logger.warning("History fetch failed: %s", e)

        logger.debug("Last user tone detected: %s", last_user_tone)
        tone_note = build_tone_guidance(last_user_tone)

        base_system_prefix = llm_prompt_prefix or (
            "You are a helpful assistant. Use the provided context to answer the question. "
            "If the answer is not present in the context, say you don't know."
        )
        
        # Add tone guidance
        enhanced_prefix = f"Conversation Tone Guidance:\n{tone_note}\n\n{base_system_prefix}"
        
        # Add personalization if profile available
        if profile or (requester and requester.get("role") != "Guest"):
            enhanced_prefix = build_personalized_prompt(
                enhanced_prefix, profile or {}, chat_history, query_text, requester or {}
            )
        
        return enhanced_prefix
    
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
        logger.info(
            "RAG_QUERY_START: query_len=%d n_results=%d use_llm=%s max_tokens=%d session_id=%s user=%s",
            len(query_text or ""), n_results, use_llm, max_tokens, session_id, (requester or {}).get("user_id", "anonymous")
        )
        logger.debug("RAG_QUERY_TEXT: %s", query_text)

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
        # DEBUG: Log final query sent to LLM
        logger.info(
            "LLM_QUERY_DEBUG: user=%s provider=%s query_len=%d context_len=%d prefix_len=%d",
            (requester or {}).get("user_id", "anonymous"),
            self.__class__.__name__,
            len(query_text),
            len(context_text or ""),
            len(final_prefix)
        )
        logger.info(
            "LLM_FINAL_PROMPT: %s\n\nCONTEXT: %s\n\nQUERY: %s",
            final_prefix,
            context_text or "",
            query_text
        )
        
        answer = await self.generate_response(
            query_text, context_text, final_prefix, use_llm, max_tokens, session_id
        )

        return self.build_base_response(
            visible_docs, filtered_result, raw_docs, raw_metadatas, raw_ids, raw_distances, context_text, answer
        )