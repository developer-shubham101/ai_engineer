from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional

router = APIRouter()

class AnalyzeRequest(BaseModel):
    text: str

class AnalyzeResponse(BaseModel):
    word_count: int
    char_count: int
    language: Optional[str] = "unknown"

@router.post("/", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    text = req.text or ""
    words = text.split()
    # Basic analysis — we will replace with real models later
    return AnalyzeResponse(word_count=len(words), char_count=len(text), language="en")