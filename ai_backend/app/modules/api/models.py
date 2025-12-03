"""API request/response models."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """RAG query request model."""
    question: str = Field(..., description="User question")
    top_k: int = Field(3, description="Number of documents to retrieve")
    use_llm: bool = Field(False, description="Whether to use LLM for response generation")
    max_tokens: int = Field(256, description="Maximum tokens for LLM response")
    temperature: float = Field(0.1, description="Temperature for LLM response")
    category: Optional[str] = Field(None, description="Document category filter")
    debug: bool = Field(False, description="Enable debug mode")
    local_llm_model: Optional[str] = Field(None, description="Specific local model to use")


class DocumentChunk(BaseModel):
    """Retrieved document chunk."""
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: float


class QueryResponse(BaseModel):
    """RAG query response model."""
    answer: Optional[str] = None
    retrieved: List[DocumentChunk] = []
    context: Optional[str] = None
    final_prompt: Optional[str] = None  # Debug mode only


class AuthRequest(BaseModel):
    """Authentication request model."""
    username: str
    password: str


class AuthResponse(BaseModel):
    """Authentication response model."""
    access_token: str
    token_type: str = "bearer"
    user: Dict[str, Any]


class DocumentRequest(BaseModel):
    """Document creation/update request."""
    text: str
    metadata: Dict[str, Any]


class DocumentResponse(BaseModel):
    """Document operation response."""
    success: bool
    message: str
    document_id: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    code: Optional[str] = None