"""Embedding provider implementations."""

import numpy as np
from typing import List
import logging
from sentence_transformers import SentenceTransformer

from .interfaces import IEmbeddingProvider
from ..config.settings import settings

logger = logging.getLogger(__name__)


class SentenceTransformerEmbedding(IEmbeddingProvider):
    """Sentence Transformer embedding provider."""
    
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.EMBEDDING_MODEL_NAME
        self.model = None
        self._dimension = None
    
    def _load_model(self):
        """Lazy load the embedding model."""
        if self.model is None:
            try:
                # Try to load from local path first
                model_path = settings.EMBEDDINGS_DIR / self.model_name
                if model_path.exists():
                    self.model = SentenceTransformer(str(model_path))
                    logger.info(f"Loaded embedding model from local path: {model_path}")
                else:
                    # Fall back to downloading from HuggingFace
                    self.model = SentenceTransformer(self.model_name)
                    logger.info(f"Loaded embedding model from HuggingFace: {self.model_name}")
                
                # Cache dimension
                self._dimension = self.model.get_sentence_embedding_dimension()
                
            except Exception as e:
                logger.error(f"Failed to load embedding model {self.model_name}: {e}")
                raise
    
    async def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        self._load_model()
        
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            logger.debug(f"Encoded {len(texts)} texts to embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to encode texts: {e}")
            raise
    
    async def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to embedding."""
        embeddings = await self.encode([text])
        return embeddings[0]
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        if self._dimension is None:
            self._load_model()
        return self._dimension
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name


class MockEmbeddingProvider(IEmbeddingProvider):
    """Mock embedding provider for testing."""
    
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.model_name = "mock-embedding-model"
    
    async def encode(self, texts: List[str]) -> np.ndarray:
        """Generate mock embeddings."""
        # Generate deterministic embeddings based on text hash
        embeddings = []
        for text in texts:
            # Simple hash-based embedding
            hash_val = hash(text)
            np.random.seed(abs(hash_val) % (2**32))
            embedding = np.random.normal(0, 1, self.dimension)
            embedding = embedding / np.linalg.norm(embedding)  # Normalize
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    async def encode_single(self, text: str) -> np.ndarray:
        """Encode single text to mock embedding."""
        embeddings = await self.encode([text])
        return embeddings[0]
    
    def get_dimension(self) -> int:
        """Get embedding dimension."""
        return self.dimension
    
    def get_model_name(self) -> str:
        """Get model name."""
        return self.model_name


class EmbeddingManager:
    """Factory for embedding providers."""
    
    _instance = None
    _provider = None
    
    @classmethod
    def get_provider(cls, provider_type: str = "sentence_transformer", **kwargs) -> IEmbeddingProvider:
        """Get embedding provider instance (singleton)."""
        if cls._provider is None:
            if provider_type == "sentence_transformer":
                cls._provider = SentenceTransformerEmbedding(**kwargs)
            elif provider_type == "mock":
                cls._provider = MockEmbeddingProvider(**kwargs)
            else:
                raise ValueError(f"Unknown embedding provider: {provider_type}")
        
        return cls._provider
    
    @classmethod
    def reset(cls):
        """Reset singleton (useful for testing)."""
        cls._provider = None