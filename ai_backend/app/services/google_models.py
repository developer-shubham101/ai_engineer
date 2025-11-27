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
# from .llm_service import IdeaRequest, IdeaResponse # Removed to fix circular import
from .prompt_builder import build_tone_guidance, build_prompt_with_selected_chunks, estimate_tokens_from_text

load_dotenv()


class ChatRequest(BaseModel):
    user_input: str = Field(..., min_length=1, description="The user's message to the chatbot.")


class ChatResponse(BaseModel):
    ai_response: str

class IdeaRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="The topic to generate ideas for.")

class IdeaResponse(BaseModel):
    ideas: str


import logging
from typing import Optional, Dict, Any

from app.services.base_rag_service import BaseRAGService
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


class GoogleRAGService(BaseRAGService):
    """
    Google RAG service implementation using Google Gemini models.
    Inherits common functionality from BaseRAGService.
    """
    
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
        Generate a response using Google Gemini LLM.
        """
        if not use_llm:
            return None
            
        if not google_llm:
            raise ConnectionError("Google LLM is not initialized. Check your API key.")

        prompt = build_prompt_with_selected_chunks(final_prefix, context_text, query_text)

        try:
            from app.logging_config import log_llm_interaction, log_sensitive_debug, log_performance_metric
            import time
            
            google_start_time = time.time()
            prompt_tokens = estimate_tokens_from_text(prompt)
            
            log_llm_interaction(
                logger, "GOOGLE_GEMINI", prompt_tokens, 0,  # response tokens unknown yet
                prompt_len=len(prompt), session_id=session_id or "none",
                model="gemini-2.5-flash"
            )
            
            log_sensitive_debug(
                logger, "Google LLM request",
                full_prompt=prompt, prompt_len=len(prompt)
            )
            
            answer = google_llm.invoke(prompt)
            answer_content = answer.content if answer and hasattr(answer, 'content') else str(answer)
            
            google_duration = (time.time() - google_start_time) * 1000
            response_tokens = estimate_tokens_from_text(answer_content)
            
            log_llm_interaction(
                logger, "GOOGLE_GEMINI", prompt_tokens, response_tokens,
                response_len=len(answer_content), duration_ms=google_duration,
                session_id=session_id or "none", model="gemini-2.5-flash"
            )
            
            log_performance_metric(
                logger, "GOOGLE_LLM_GENERATION", google_duration,
                prompt_tokens=prompt_tokens, response_tokens=response_tokens,
                session_id=session_id
            )
            
            log_sensitive_debug(
                logger, "Google LLM response",
                response_text=answer_content, response_len=len(answer_content)
            )
            
            return answer_content
        except Exception as e:
            logger.exception("Google LLM call failed: %s", e)
            raise


# Create global instance
_google_rag_service = GoogleRAGService()


async def query_google_rag(
        query_text: str,
        n_results: int = 3,
        requester: Optional[Dict[str, str]] = None,
        llm_prompt_prefix: Optional[str] = None,
        use_llm: bool = True,
        session_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Query the Google RAG service using the base RAG functionality.
    """
    return await _google_rag_service.query_rag(
        query_text=query_text,
        n_results=n_results,
        requester=requester,
        llm_prompt_prefix=llm_prompt_prefix,
        use_llm=use_llm,
        max_tokens=256,  # Google doesn't use max_tokens in the same way
        session_id=session_id
    )


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
