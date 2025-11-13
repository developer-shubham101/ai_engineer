# app/services/huggingface_api_models.py
# This file contains the logic for interacting with the Hugging Face Inference API.

import os
from dotenv import load_dotenv
from langchain.llms import HuggingFaceEndpoint # Note: Using legacy llms wrapper for stability
from .llm_service import TextRequest, GenerationResponse

# Load environment variables from the .env file
load_dotenv()

# --- LangChain LLM Initialization for Hugging Face ---
try:
    # Get the API token from the environment
    hf_api_token = os.environ.get("HUGGING_FACE_HUB_API_TOKEN")

    if not hf_api_token:
        print("Warning: Hugging Face API token not found. HF endpoint will be unavailable.")
        hf_llm = None
    else:
        # Initialize the LLM object, which represents our connection.
        # We point it to a specific, powerful model on the Hub.
        # Mistral-7B-Instruct is a great, reliable choice.
        hf_llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            max_new_tokens=256,
            temperature=0.7,
            huggingfacehub_api_token=hf_api_token,
        )

except Exception as e:
    print(f"Warning: Could not initialize HuggingFaceEndpoint. Error: {e}")
    hf_llm = None


def generate_text_hf(request: TextRequest) -> GenerationResponse:
    """Performs text generation using the Hugging Face Inference API via LangChain."""
    if not hf_llm:
        raise ConnectionError("Hugging Face LLM is not initialized. Check your API key and model setup.")

    try:
        # With LangChain, calling the LLM is as simple as passing the text.
        # LangChain handles the API call, headers, and parsing the response.
        result_text = hf_llm(request.text)
        return GenerationResponse(generated_text=result_text)
    except Exception as e:
        # LangChain will raise exceptions on API errors, which we can catch.
        raise ConnectionError(f"Failed to get response from Hugging Face API: {e}")