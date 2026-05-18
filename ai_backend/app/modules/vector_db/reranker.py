"""Cross-encoder reranker for improved retrieval quality.

This module implements a reranking layer that improves upon vector similarity
by using a cross-encoder model to score query-document pairs.
"""
import logging
from typing import List, Dict, Any, Optional
from sentence_transformers import CrossEncoder
import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Cross-encoder reranker for document retrieval.
    
    Uses cross-encoder/ms-marco-MiniLM-L6-v2 to rerank documents based on
    query-document relevance scores, improving upon cosine similarity.
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"):
        """Initialize the cross-encoder reranker.
        
        Args:
            model_name: HuggingFace model name for cross-encoder
        """
        self.model_name = model_name
        self._model: Optional[CrossEncoder] = None
        logger.info(f"CrossEncoderReranker initialized with model: {model_name}")
    
    def _load_model(self) -> CrossEncoder:
        """Lazy load the cross-encoder model."""
        if self._model is None:
            logger.info(f"Loading cross-encoder model: {self.model_name}")
            try:
                self._model = CrossEncoder(self.model_name)
                logger.info("Cross-encoder model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load cross-encoder model: {e}")
                raise
        return self._model
    
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Rerank documents using cross-encoder scores.
        
        Args:
            query: User query text
            documents: List of retrieved documents with 'text' field
            top_k: Number of top documents to return after reranking
            
        Returns:
            Reranked list of documents with updated scores
        """
        if not documents:
            logger.warning("No documents to rerank")
            return []
        
        if len(documents) <= top_k:
            logger.info(f"Document count ({len(documents)}) <= top_k ({top_k}), returning all")
            return documents
        
        try:
            # Load model
            model = self._load_model()
            
            # Prepare query-document pairs
            pairs = [[query, doc.get("text", "")] for doc in documents]
            
            # Get cross-encoder scores
            logger.info(f"Reranking {len(documents)} documents with cross-encoder")
            scores = model.predict(pairs)
            
            # Add scores to documents
            for doc, score in zip(documents, scores):
                doc["rerank_score"] = float(score)
                doc["original_distance"] = doc.get("distance", 0.0)
            
            # Sort by rerank score (higher is better)
            reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
            
            # Return top_k
            top_docs = reranked[:top_k]
            
            logger.info(f"Reranking complete. Top score: {top_docs[0]['rerank_score']:.4f}, "
                       f"Bottom score: {top_docs[-1]['rerank_score']:.4f}")
            
            return top_docs
            
        except Exception as e:
            logger.error(f"Reranking failed: {e}, returning original documents")
            return documents[:top_k]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "model_loaded": self._model is not None,
            "model_type": "cross-encoder"
        }


class RerankerFactory:
    """Factory for creating reranker instances."""
    
    @staticmethod
    def create_reranker(
        reranker_type: str = "cross-encoder",
        model_name: Optional[str] = None
    ) -> CrossEncoderReranker:
        """Create a reranker instance.
        
        Args:
            reranker_type: Type of reranker (currently only 'cross-encoder')
            model_name: Optional model name override
            
        Returns:
            Reranker instance
        """
        if reranker_type == "cross-encoder":
            model = model_name or "cross-encoder/ms-marco-MiniLM-L6-v2"
            return CrossEncoderReranker(model_name=model)
        else:
            raise ValueError(f"Unknown reranker type: {reranker_type}")
