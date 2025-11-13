# app/services/local_models.py

from transformers import pipeline
# Import the shared Pydantic models from our central service file
from .llm_service import TextRequest, SummarizationResponse, GenerationResponse, SentimentResponse

print("Loading local models... This may take a moment.")

summarizer = pipeline("summarization", model="google/flan-t5-small")
generator = pipeline("text-generation", model="google/flan-t5-small")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

print("Local models loaded successfully.")

def summarize_text(request: TextRequest) -> SummarizationResponse:
    result = summarizer(request.text, max_length=150, min_length=30, do_sample=False)
    return SummarizationResponse(summary_text=result[0]['summary_text'])

def generate_text(request: TextRequest) -> GenerationResponse:
    result = generator(request.text, max_length=50)
    return GenerationResponse(generated_text=result[0]['generated_text'])

def classify_sentiment(request: TextRequest) -> SentimentResponse:
    result = sentiment_analyzer(request.text)
    return SentimentResponse(label=result[0]['label'], score=result[0]['score'])