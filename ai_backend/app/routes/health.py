from fastapi import APIRouter
from pydantic import BaseModel
import platform

router = APIRouter()

class HealthResp(BaseModel):
    status: str
    python: str

@router.get("/", response_model=HealthResp)
async def health():
    return {"status": "ok", "python": platform.python_version()}