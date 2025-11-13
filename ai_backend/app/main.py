# app/main.py

from fastapi import FastAPI, HTTPException
from .services import llm_service

from .services import rag_service # <-- ADD THIS IMPORT

# # Initialize the FastAPI app
# app = FastAPI(
#     title="AI Engineering API",
#     description="A foundational API for AI engineering skills development.",
#     version="1.0.0",
# )

# --- Add this lifespan event handler ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # This code runs on startup
    print("Application startup...")
    rag_service.initialize_rag_chain()
    yield
    # This code runs on shutdown (not used here, but good practice)
    print("Application shutdown...")

# Update your FastAPI app initialization
app = FastAPI(
    title="AI Engineering API",
    description="A foundational API for AI engineering skills development.",
    version="1.0.0",
    lifespan=lifespan # <-- ADD THIS
)

# --- API Endpoints ---

@app.get("/", tags=["General"])
def read_root():
    """A simple health check endpoint."""
    return {"status": "ok", "message": "Welcome to the AI Engineering API!"}

# --- LLM Endpoints ---

@app.post("/summarize",
          response_model=llm_service.SummarizationResponse,
          tags=["LLM Services"])
def summarize(request: llm_service.TextRequest):
    """
    Endpoint to summarize a given piece of text.
    """
    try:
        return llm_service.summarize_text(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate",
          response_model=llm_service.GenerationResponse,
          tags=["LLM Services"])
def generate(request: llm_service.TextRequest):
    """
    Endpoint to generate text based on a given prompt.
    """
    try:
        return llm_service.generate_text(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sentiment",
          response_model=llm_service.SentimentResponse,
          tags=["LLM Services"])
def sentiment(request: llm_service.TextRequest):
    """
    Endpoint to analyze the sentiment of a given piece of text.
    """
    try:
        return llm_service.classify_sentiment(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/generate/openai",
          response_model=llm_service.GenerationResponse,
          tags=["LLM Services (OpenAI)"])
def generate_openai(request: llm_service.TextRequest):
    """
    Endpoint to generate text using OpenAI's gpt-3.5-turbo.
    """
    try:
        return llm_service.generate_text_openai(request)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# @app.post("/generate/hf",
#           response_model=llm_service.GenerationResponse,
#           tags=["LLM Services (Hugging Face API)"])
# def generate_hf(request: llm_service.TextRequest):
#     """
#     Endpoint to generate text using the Hugging Face Inference API.
#     Uses a model like Mistral-7B.
#     """
#     try:
#         return llm_service.generate_text_hf_inference(request)
#     except ConnectionError as e:
#         raise HTTPException(status_code=503, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/hf",
          response_model=llm_service.GenerationResponse,
          tags=["LLM Services (Hugging Face API)"])
def generate_hf(request: llm_service.TextRequest):
    """
    Endpoint to generate text using the Hugging Face Inference API via LangChain.
    Uses a model like Gemma-7B.
    """
    try:
        # Call the NEW langchain function
        return llm_service.generate_text_hf_inference_langchain(request)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/ideas",
          response_model=llm_service.IdeaResponse,
          tags=["LLM Services (LangChain)"])
def generate_ideas(request: llm_service.IdeaRequest):
    """
    Endpoint to generate content ideas using a LangChain chain.
    """
    try:
        # This line calls the function in our service file.
        return llm_service.generate_content_ideas(request)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat",
          response_model=llm_service.ChatResponse,
          tags=["LLM Services (Conversational)"])
def chat(request: llm_service.ChatRequest):
    """
    Endpoint for a conversational chat with memory.
    """
    try:
        return llm_service.get_chat_response(request)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- NEW RAG Endpoint ---
@app.post("/ask-document",
          response_model=llm_service.GenerationResponse, # We can reuse this model
          tags=["RAG Services"])
def ask_document(request: llm_service.TextRequest):
    """
    Endpoint to ask questions about a pre-loaded document.
    """
    try:
        answer = rag_service.query_document(request)
        return llm_service.GenerationResponse(generated_text=answer)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))