# app/services/llm_service.py
# Legacy service file - kept for backward compatibility with existing endpoints
# New multi-provider RAG architecture is in base_rag_service.py and provider-specific services

from pydantic import BaseModel, Field

# --- Shared Pydantic Models ---
# These are used by legacy endpoints in main.py

class TextRequest(BaseModel):
    """Request model for endpoints taking a single text input."""
    text: str = Field(..., min_length=1, description="Text to process.")

class SummarizationResponse(BaseModel):
    summary_text: str

class GenerationResponse(BaseModel):
    generated_text: str

class SentimentResponse(BaseModel):
    label: str
    score: float

# Import models from google_models for backward compatibility
from .google_models import ChatRequest, ChatResponse, IdeaRequest, IdeaResponse
from .google_models import generate_content_ideas, get_chat_response

# Placeholder functions for removed services - these will raise NotImplementedError
def summarize_text(request: TextRequest) -> SummarizationResponse:
    raise NotImplementedError("Local summarization service removed. Use multi-provider RAG endpoints instead.")

def generate_text(request: TextRequest) -> GenerationResponse:
    raise NotImplementedError("Local generation service removed. Use multi-provider RAG endpoints instead.")

def classify_sentiment(request: TextRequest) -> SentimentResponse:
    raise NotImplementedError("Local sentiment service removed. Use /api/rag/sentiment endpoint instead.")

def generate_text_openai(request: TextRequest) -> GenerationResponse:
    raise NotImplementedError("OpenAI service removed. Use /api/rag/gpt/query endpoint instead.")

def generate_text_hf_inference_langchain(request: TextRequest) -> GenerationResponse:
    raise NotImplementedError("HuggingFace service removed. Use /api/rag/hf/query endpoint instead.")