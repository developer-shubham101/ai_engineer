"""Middleware pattern for RAG request/response processing."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
import time
import logging
from .interfaces import RAGRequest, RAGResponse

logger = logging.getLogger(__name__)


class RAGMiddleware(ABC):
    """Abstract RAG middleware."""
    
    @abstractmethod
    async def process_request(self, request: RAGRequest) -> RAGRequest:
        """Process incoming request."""
        pass
    
    @abstractmethod
    async def process_response(self, request: RAGRequest, response: RAGResponse) -> RAGResponse:
        """Process outgoing response."""
        pass


class LoggingMiddleware(RAGMiddleware):
    """Logs requests and responses."""
    
    async def process_request(self, request: RAGRequest) -> RAGRequest:
        logger.info(f"RAG request: provider={request.provider}, question_length={len(request.question)}")
        return request
    
    async def process_response(self, request: RAGRequest, response: RAGResponse) -> RAGResponse:
        doc_count = len(response.retrieved_documents)
        has_answer = bool(response.answer)
        logger.info(f"RAG response: docs={doc_count}, has_answer={has_answer}")
        return response


class TimingMiddleware(RAGMiddleware):
    """Tracks request timing."""
    
    def __init__(self):
        self._start_times: Dict[str, float] = {}
    
    async def process_request(self, request: RAGRequest) -> RAGRequest:
        request_id = id(request)
        self._start_times[request_id] = time.time()
        return request
    
    async def process_response(self, request: RAGRequest, response: RAGResponse) -> RAGResponse:
        request_id = id(request)
        if request_id in self._start_times:
            duration = time.time() - self._start_times[request_id]
            response.metadata = response.metadata or {}
            response.metadata["processing_time"] = duration
            del self._start_times[request_id]
            logger.info(f"RAG processing took {duration:.2f}s")
        return response


class SecurityMiddleware(RAGMiddleware):
    """Applies security filtering."""
    
    async def process_request(self, request: RAGRequest) -> RAGRequest:
        # Sanitize input
        if request.question:
            request.question = request.question.strip()[:1000]  # Limit length
        return request
    
    async def process_response(self, request: RAGRequest, response: RAGResponse) -> RAGResponse:
        # Filter sensitive information from response
        if response.answer and request.user:
            user_role = request.user.get("role", "Guest")
            if user_role in ["Guest", "Employee"]:
                # Remove potential sensitive patterns
                sensitive_patterns = ["password", "secret", "key", "token"]
                answer = response.answer
                for pattern in sensitive_patterns:
                    if pattern.lower() in answer.lower():
                        logger.warning(f"Potential sensitive info filtered for role {user_role}")
                        break
        return response


class CachingMiddleware(RAGMiddleware):
    """Simple response caching."""
    
    def __init__(self):
        self._cache: Dict[str, RAGResponse] = {}
        self._max_cache_size = 100
    
    def _get_cache_key(self, request: RAGRequest) -> str:
        """Generate cache key from request."""
        return f"{request.provider}:{hash(request.question)}:{request.top_k}"
    
    async def process_request(self, request: RAGRequest) -> RAGRequest:
        cache_key = self._get_cache_key(request)
        if cache_key in self._cache:
            # Mark as cached for response processing
            request.metadata = request.metadata or {}
            request.metadata["_cached_response"] = self._cache[cache_key]
        return request
    
    async def process_response(self, request: RAGRequest, response: RAGResponse) -> RAGResponse:
        # Return cached response if available
        if request.metadata and "_cached_response" in request.metadata:
            cached_response = request.metadata["_cached_response"]
            cached_response.metadata = cached_response.metadata or {}
            cached_response.metadata["from_cache"] = True
            return cached_response
        
        # Cache new response
        cache_key = self._get_cache_key(request)
        if len(self._cache) >= self._max_cache_size:
            # Simple LRU: remove first item
            self._cache.pop(next(iter(self._cache)))
        
        self._cache[cache_key] = response
        return response


class MiddlewareStack:
    """Manages middleware stack."""
    
    def __init__(self):
        self._middlewares: List[RAGMiddleware] = []
    
    def add(self, middleware: RAGMiddleware):
        """Add middleware to stack."""
        self._middlewares.append(middleware)
    
    def remove(self, middleware_type: type):
        """Remove middleware by type."""
        self._middlewares = [m for m in self._middlewares if not isinstance(m, middleware_type)]
    
    async def process_request(self, request: RAGRequest) -> RAGRequest:
        """Process request through all middlewares."""
        for middleware in self._middlewares:
            request = await middleware.process_request(request)
        return request
    
    async def process_response(self, request: RAGRequest, response: RAGResponse) -> RAGResponse:
        """Process response through all middlewares (in reverse order)."""
        for middleware in reversed(self._middlewares):
            response = await middleware.process_response(request, response)
        return response


def create_default_middleware_stack() -> MiddlewareStack:
    """Create default middleware stack."""
    stack = MiddlewareStack()
    stack.add(LoggingMiddleware())
    stack.add(TimingMiddleware())
    stack.add(SecurityMiddleware())
    stack.add(CachingMiddleware())
    return stack