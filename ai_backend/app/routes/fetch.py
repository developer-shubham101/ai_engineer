from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.utils.http_client import fetch_json

router = APIRouter()

class FetchRequest(BaseModel):
    url: str

class FetchResponse(BaseModel):
    url: str
    status: int
    body_snippet: str

@router.post("/", response_model=FetchResponse)
async def fetch(req: FetchRequest):
    try:
        status, text = await fetch_json(req.url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    snippet = text[:500]
    return FetchResponse(url=req.url, status=status, body_snippet=snippet)