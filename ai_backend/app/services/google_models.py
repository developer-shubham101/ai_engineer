# app/services/google_models.py

import os

from dotenv import load_dotenv
from langchain.chains import LLMChain, ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# We need a new Pydantic model for our chat endpoint
from pydantic import BaseModel, Field

from .chroma_utils import ensure_chroma_client, query_collection
from .llm_service import IdeaRequest, IdeaResponse
from .prompt_builder import build_tone_guidance
from .rag_local_service import estimate_tokens_from_text, build_prompt_with_selected_chunks

load_dotenv()


class ChatRequest(BaseModel):
    user_input: str = Field(..., min_length=1, description="The user's message to the chatbot.")


class ChatResponse(BaseModel):
    ai_response: str


import logging
from typing import Optional, Dict, Any

from app.services.support_chat import fetch_recent_messages
from app.services.utility import embed_texts, DEFAULT_PERSIST_DIR, DEFAULT_COLLECTION_NAME

logger = logging.getLogger(__name__)

# The definitive list of Google's flagship generative AI models, ordered by size/capability.
# Smallest / Fastest / Most Cost-Effective ----------------------> Largest / Most Capable
google_ai_models = [
    # ----------------------------------------------------------------------------------
    # 1. Nano Family (Smallest, On-Device)
    # ----------------------------------------------------------------------------------
    "gemini-nano",  # Smallest, most efficient model. Primarily designed for
    # running directly on mobile devices (e.g., Pixel phones).
    # Not typically available via the cloud API.

    # ----------------------------------------------------------------------------------
    # 2. Flash-Lite Family (High-Throughput, Low-Cost)
    # ----------------------------------------------------------------------------------
    "gemini-2.5-flash-lite",  # The fastest and most cost-optimized cloud model.
    # Excellent for high-volume, low-latency tasks like simple
    # classification, short summaries, or chat.

    # ----------------------------------------------------------------------------------
    # 3. Flash Family (Speed and General Capability)
    # ----------------------------------------------------------------------------------
    "gemini-2.5-flash",  # The default choice for most web and general applications.
    # Offers a great balance of speed, capability, and price.
    # Highly effective for summarization, data extraction, and
    # multi-turn chat.

    # Note: gemini-2.0-flash is a capable but older generation model, largely superseded
    #       by the 2.5 series due to better performance and features.

    # ----------------------------------------------------------------------------------
    # 4. Pro Family (Most Capable, Complex Reasoning)
    # ----------------------------------------------------------------------------------
    "gemini-2.5-pro",  # The state-of-the-art model for complex tasks.
    # Excels at advanced reasoning, coding, deep data analysis,
    # and highly nuanced, multi-step problem-solving.
    # Best for high-quality, complex enterprise use cases.

    # Note: A new, most powerful version, 'gemini-3-pro' is available in Preview and is
    #       currently the most advanced model in the family.
]

try:
    google_llm = ChatGoogleGenerativeAI(
        model=google_ai_models[1],
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
        convert_system_message_to_human=True
    )

    idea_template = """
    You are an expert content strategist.
    Your goal is to generate 5 engaging blog post titles for the following topic.
    Topic: {topic}
    Provide the titles as a numbered list.
    """

    prompt = PromptTemplate(
        input_variables=["topic"],
        template=idea_template
    )

    idea_chain = LLMChain(llm=google_llm, prompt=prompt)

except Exception as e:
    logger.warning(f"Could not initialize Google legacy chain. Error: {e}")
    idea_chain = None


def generate_content_ideas(request: IdeaRequest) -> IdeaResponse:
    if not idea_chain:
        raise ConnectionError("Google Legacy Chain is not initialized. Check your API key.")

    try:
        result = idea_chain.invoke({"topic": request.topic})
        return IdeaResponse(ideas=result['text'])
    except Exception as e:
        raise ConnectionError(f"Failed to get response from Google legacy chain: {e}")


# --- NEW: LangChain Conversational Chain with Memory ---

try:
    # We create a single, shared memory object for this simple example.
    # In a real multi-user app, you'd manage one memory object per user session.
    chat_memory = ConversationBufferMemory()

    # The ConversationChain is simpler than LLMChain; it has a default prompt.
    conversation_chain = ConversationChain(
        llm=google_llm,
        memory=chat_memory,
        verbose=True  # Set to True to see the full prompt being sent to the LLM in your terminal
    )

except Exception as e:
    logger.warning(f"Could not initialize Google conversation chain. Error: {e}")
    conversation_chain = None


def get_chat_response(request: ChatRequest) -> ChatResponse:
    """Generates a conversational response using a chain with memory."""
    if not conversation_chain:
        raise ConnectionError("Google Conversation Chain is not initialized.")

    try:
        # The chain's 'predict' method takes the user input and returns the AI's response.
        # It automatically handles loading history, formatting the prompt, and storing the new turn.
        ai_message = conversation_chain.predict(input=request.user_input)
        return ChatResponse(ai_response=ai_message)

    except Exception as e:
        raise ConnectionError(f"Failed to get response from Google conversation chain: {e}")


async def query_google_rag(
        query_text: str,
        n_results: int = 3,
        requester: Optional[Dict[str, str]] = None,
        llm_prompt_prefix: Optional[str] = None,
        use_llm: bool = True,
        session_id: Optional[str] = None
) -> Dict[str, Any]:
    # Ensure Chroma client
    client, collection = ensure_chroma_client(
        persist_directory=str(DEFAULT_PERSIST_DIR),
        collection_name=DEFAULT_COLLECTION_NAME
    )

    logger.debug(
        "query_google_rag called: query_text_len=%d n_results=%d use_llm=%s session_id=%s requester=%s",
        len(query_text or ""), n_results, use_llm, session_id, (requester or {}).get("user_id"))

    if not query_text:
        raise ValueError("query_text must be provided")

    # 1. Get embedding for query
    try:
        q_emb = (await embed_texts([query_text]))[0]
        logger.debug("Computed query embedding.")
    except Exception as e:
        logger.exception("Failed to embed query: %s", e)
        raise

    # 2. Retrieve from Chroma
    try:
        result = query_collection(collection=collection, query_embeddings=[q_emb], n_results=n_results)
    except Exception:
        result = query_collection(collection=collection, query_texts=[query_text], n_results=n_results)

    raw_docs = (result.get("documents") or [[]])[0]
    raw_metadatas = (result.get("metadatas") or [[]])[0]
    raw_ids = (result.get("ids") or [[]])[0]
    raw_distances = (result.get("distances") or [[]])[0]

    logger.debug("Raw retrieval counts: docs=%d metadatas=%d ids=%d distances=%d",
                 len(raw_docs), len(raw_metadatas), len(raw_ids), len(raw_distances))

    # 3. RBAC filtering
    def _allowed_by_metadata(meta: Optional[Dict[str, Any]], requester: Optional[Dict[str, str]]) -> bool:
        sens = meta.get("sensitivity", "public_internal") if meta else "public_internal"
        if sens == "personal":
            owner = meta.get("owner_id")
            if requester and owner == requester.get("user_id"): return True
            return requester and requester.get("role") in ("HR", "Legal", "Executive")
        if sens == "highly_confidential":
            return requester and requester.get("role") in ("Legal", "Executive")
        if sens == "role_confidential":
            allowed_roles = meta.get("allowed_roles") or []
            if requester and requester.get("role") in allowed_roles: return True
            return requester and requester.get("role") in ("HR", "Legal", "Executive")
        if sens == "department_confidential":
            if requester and requester.get("department") == meta.get("department"): return True
            return requester and requester.get("role") in ("HR", "Legal", "Executive")
        return True

    visible_docs, visible_metas, visible_ids, visible_distances = [], [], [], []
    public_summaries, filtered_details = [], []
    filtered_out_count = 0

    for doc, meta, id_, dist in zip(raw_docs, raw_metadatas, raw_ids, raw_distances):
        if _allowed_by_metadata(meta, requester):
            visible_docs.append(doc)
            visible_metas.append(meta)
            visible_ids.append(id_)
            visible_distances.append(dist)
        else:
            filtered_out_count += 1
            ps = meta.get("public_summary") if isinstance(meta, dict) else None
            if ps: public_summaries.append(ps)
            filtered_details.append(
                {"id": id_, "sensitivity": meta.get("sensitivity"), "department": meta.get("department"),
                 "source": meta.get("source")})

    # 4. Build Context
    context_text = "\n\n---\n\n".join(visible_docs or [])
    logger.info("Post-filtering: visible_docs=%d filtered_out=%d", len(visible_docs), filtered_out_count)

    out = {
        "documents": visible_docs, "metadatas": visible_metas, "ids": visible_ids, "distances": visible_distances,
        "raw_documents": raw_docs, "raw_metadatas": raw_metadatas, "raw_ids": raw_ids, "raw_distances": raw_distances,
        "context": context_text, "filtered_out_count": filtered_out_count, "public_summaries": public_summaries,
        "filtered_details": filtered_details,
    }

    # 5. Tone-Based Prefix Injection
    last_user_tone = None
    if session_id:
        try:
            history = fetch_recent_messages(session_id, limit=10)
            for m in reversed(history):
                if m.get("speaker") == "user" and m.get("tone"):
                    last_user_tone = m["tone"]
                    break
        except Exception as e:
            logger.warning("Tone fetch failed: %s", e)

    tone_note = build_tone_guidance(last_user_tone)
    system_prefix = llm_prompt_prefix or (
        "You are a helpful assistant. Use the provided context to answer the question. "
        "If the answer is not present in the context, say you don't know."
    )
    final_prefix = f"Conversation Tone Guidance:\n{tone_note}\n\n{system_prefix}"
    approx_prefix_tokens = estimate_tokens_from_text(final_prefix)
    approx_context_tokens = estimate_tokens_from_text(context_text)
    logger.debug("Prompt sizes: prefix_chars=%d context_chars=%d est_prefix_tokens=%d est_context_tokens=%d",
                 len(final_prefix), len(context_text), approx_prefix_tokens, approx_context_tokens)

    # 6. LLM Call
    if use_llm:
        if not google_llm:
            raise ConnectionError("Google LLM is not initialized. Check your API key.")

        prompt = build_prompt_with_selected_chunks(final_prefix, context_text, query_text)

        try:
            answer = google_llm.invoke(prompt)
            answer_len = len(answer.content) if answer and hasattr(answer, 'content') else 0
            logger.info("Google LLM returned answer (approx length=%d) for query session=%s", answer_len, session_id)
            out["answer"] = answer.content
        except Exception as e:
            logger.exception("Google LLM call failed: %s", e)
            raise

    return out


async def generate_content_ideas_with_rag(request: IdeaRequest, requester: Dict[str, str]) -> IdeaResponse:
    """
    Generate content ideas using RAG.
    This function uses the query_google_rag function to get context from the knowledge base
    and then uses the Google LLM to generate blog post titles.
    """
    rag_result = await query_google_rag(
        query_text=request.topic,
        n_results=5,
        requester=requester,
        use_llm=False  # We will call the LLM separately
    )

    context = rag_result.get("context", "")

    # Now, call the LLM with the context
    if not idea_chain:
        raise ConnectionError("Google Legacy Chain is not initialized. Check your API key.")

    try:
        # Modify the prompt to include the context
        idea_template_with_rag = """
        You are an expert content strategist.
        Your goal is to generate 5 engaging blog post titles for the following topic,
        using the provided context to inspire the titles.

        Context:
        {context}

        Topic: {topic}

        Provide the titles as a numbered list.
        """
        prompt_with_rag = PromptTemplate(
            input_variables=["topic", "context"],
            template=idea_template_with_rag
        )
        idea_chain_with_rag = LLMChain(llm=google_llm, prompt=prompt_with_rag)
        result = idea_chain_with_rag.invoke({"topic": request.topic, "context": context})
        return IdeaResponse(ideas=result['text'])
    except Exception as e:
        raise ConnectionError(f"Failed to get response from Google legacy chain with RAG: {e}")
