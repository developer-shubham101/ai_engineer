# app/services/llm_service.py
# This file is now the central hub or "facade" for all AI services.
# It defines the shared data models and imports the actual logic from other files.

from pydantic import BaseModel, Field

# --- Shared Pydantic Models ---
# These are used by multiple services, so they live here.

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

class IdeaRequest(BaseModel):
    topic: str = Field(..., min_length=3, description="The topic to generate ideas for.")

class IdeaResponse(BaseModel):
    ideas: str

# --- Importing and Exposing Service Functions ---
# We import the functions from our specialized modules so that main.py
# can access them all from this single 'llm_service' module.

# From our local, free models
from .local_models import summarize_text, generate_text, classify_sentiment

# From our direct OpenAI integration
from .openai_models import generate_text_openai


# Models for the new Chat service
from .google_models import ChatRequest, ChatResponse

# From our stable, working Google Gemini chain
from .google_models import generate_content_ideas, get_chat_response

# You can uncomment the line below if you get the Hugging Face API working
# from .huggingface_api_models import generate_text_hf_inference_langchain