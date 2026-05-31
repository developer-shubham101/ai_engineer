"""BM25 keyword-based retrieval for hybrid search."""
from __future__ import annotations

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None

logger = logging.getLogger(__name__)


@dataclass
class BM25Document:
    """Document representation for BM25 indexing."""
    id: str
    text: str
    metadata: Dict[str, Any]
    tokens: List[str]


class BM25Index:
    """In-memory BM25 index for keyword-based retrieval."""
    
    def __init__(self):
        self.documents: List[BM25Document] = []
        self.bm25: Optional[BM25Okapi] = None
        self._initialized = False
        
        if BM25Okapi is None:
            logger.warning("rank_bm25 not installed. BM25 retrieval disabled.")
    
    def is_available(self) -> bool:
        """Check if BM25 is available."""
        return BM25Okapi is not None
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to BM25 index."""
        if not self.is_available():
            return
        
        self.documents = []
        tokenized_corpus = []
        
        for doc in documents:
            text = doc.get("text", "")
            tokens = self._tokenize(text)
            
            bm25_doc = BM25Document(
                id=doc.get("id", ""),
                text=text,
                metadata=doc.get("metadata", {}),
                tokens=tokens
            )
            self.documents.append(bm25_doc)
            tokenized_corpus.append(tokens)
        
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            self._initialized = True
            logger.info(f"BM25 index built with {len(self.documents)} documents")
    
    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Search using BM25 scoring."""
        if not self._initialized or not self.bm25:
            return []
        
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return documents with positive scores
                doc = self.documents[idx]
                results.append({
                    "id": doc.id,
                    "text": doc.text,
                    "metadata": doc.metadata,
                    "bm25_score": float(scores[idx])
                })
        
        return results
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize on whitespace and separators to handle identifiers like PTO-2024-Q1."""
        return re.findall(r'[a-z0-9]+', text.lower())
    
    def clear(self) -> None:
        """Clear the index."""
        self.documents = []
        self.bm25 = None
        self._initialized = False
