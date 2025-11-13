# app/services/openai_models.py

import os
from openai import OpenAI
from dotenv import load_dotenv
from .llm_service import TextRequest, GenerationResponse

load_dotenv()

try:
    openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
except Exception as e:
    print(f"Warning: OpenAI client could not be initialized. Error: {e}")
    openai_client = None

def generate_text_openai(request: TextRequest) -> GenerationResponse:
    if not openai_client:
        raise ConnectionError("OpenAI client is not initialized. Check your API key.")

    chat_completion = openai_client.chat.completions.create(
        messages=[{"role": "user", "content": request.text}],
        model="gpt-3.5-turbo",
    )
    generated_text = chat_completion.choices[0].message.content
    return GenerationResponse(generated_text=generated_text)