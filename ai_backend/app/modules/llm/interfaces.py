"""LLM provider interfaces."""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass


@dataclass
class LLMResponse:
    """LLM response data structure."""
    text: str
    metadata: Dict[str, Any]
    usage: Optional[Dict[str, Any]] = None
    finish_reason: Optional[str] = None


@dataclass
class RAGRequest:
    """RAG request data structure."""
    question: str
    user: Dict[str, Any]
    session_id: Optional[str] = None
    conversation_id: str = None
    top_k: int = 3
    use_llm: bool = True
    use_documents: bool = True  # Flag to control document retrieval
    max_tokens: int = 256
    temperature: float = 0.1  # Temperature controls randomness. Low = accurate and predictable. High = creative and unpredictable.
    category: Optional[str] = None
    debug: bool = False
    provider: str = "local"
    provider_specific: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None  # not coming from front end
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RetrievedDocument:
    """Retrieved document structure."""
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None


@dataclass
class RAGResponse:
    """RAG response data structure."""
    answer: Optional[str] = None
    retrieved_documents: List[RetrievedDocument] = None
    context: Optional[str] = None
    final_prompt: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.retrieved_documents is None:
            self.retrieved_documents = []
        if self.metadata is None:
            self.metadata = {}


class ILLMProvider(ABC):
    """Interface for LLM providers."""
    
    @abstractmethod
    async def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.1, **kwargs) -> LLMResponse:
        """Generate response from prompt."""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get provider name."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get current model name."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available."""
        pass
    
    @abstractmethod
    def get_max_context_length(self) -> int:
        """Get maximum context length."""
        pass


class IPromptManager(ABC):
    """Interface for prompt management."""
    
    @abstractmethod
    async def build_system_prompt(self, user: Dict[str, Any], context: str, category: Optional[str] = None) -> str:
        """Build system prompt."""
        pass
    
    @abstractmethod
    async def build_user_prompt(self, question: str, context: str) -> str:
        """Build user prompt."""
        pass
    
    @abstractmethod
    async def build_full_prompt(self, system_prompt: str, user_prompt: str, conversation_history: List[Dict[str, Any]] = None) -> str:
        """Build complete prompt."""
        pass
    
    @abstractmethod
    def truncate_context(self, context: str, max_tokens: int) -> str:
        """Truncate context to fit token limit."""
        pass
    
    @abstractmethod
    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        pass


class IRAGOrchestrator(ABC):
    """Interface for RAG orchestration."""
    
    @abstractmethod
    async def process_query(self, request: RAGRequest) -> RAGResponse:
        """Process RAG query end-to-end."""
        pass
    
    @abstractmethod
    async def retrieve_documents(self, query: str, user: Dict[str, Any], top_k: int = 5, category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Retrieve relevant documents."""
        pass
    
    @abstractmethod
    async def generate_response(self, prompt: str, provider, max_tokens: int = 256, temperature: float = 0.1) -> Optional[str]:
        """Generate response using LLM."""
        pass
    
    @abstractmethod
    async def build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context from retrieved documents."""
        pass