"""Semantic tool-call cache.

Instead of exact key matching (`tool_name:json(args)`), this cache embeds
the canonical call string and returns a cached result when a new call is
cosine-similar enough to a stored one.

Usage (drop-in replacement for the plain dict cache):
    cache = SemanticCache(threshold=0.97)
    result = cache.get(tool_name, args)   # None on miss
    cache.set(tool_name, args, result)    # store after execution
"""
from __future__ import annotations

import json
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Cosine similarity threshold — calls above this are considered equivalent.
# 0.97 is tight enough to avoid false positives for tool calls where small
# arg differences (e.g. different city names) must NOT share results.
DEFAULT_THRESHOLD = 0.97


def _canonical(tool_name: str, args: Dict[str, Any]) -> str:
    """Stable string representation of a tool call for embedding."""
    return f"{tool_name} {json.dumps(args, sort_keys=True, default=str)}"


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class SemanticCache:
    """Embedding-based cache for tool call results.

    Each entry stores:
        - the canonical call string
        - its embedding vector
        - the result

    On lookup, the query embedding is compared against all stored vectors.
    If the best match exceeds `threshold`, the stored result is returned.
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD) -> None:
        self.threshold = threshold
        self._entries: List[Tuple[np.ndarray, Any]] = []  # (embedding, result)
        self._model: Any = None  # lazy — loaded on first use

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _get_model(self):
        """Lazy-load the singleton embedding model."""
        if self._model is None:
            try:
                from app.modules.vector_db.embedding_manager import EmbeddingManager
                self._model = EmbeddingManager()
            except Exception as e:
                logger.warning("SemanticCache: could not load EmbeddingManager: %s", e)
        return self._model

    def _embed_sync(self, text: str) -> Optional[np.ndarray]:
        """Return embedding as numpy array, or None on failure."""
        model = self._get_model()
        if model is None:
            return None
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — use run_in_executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(model._model.encode, [text], False)
                    vectors = future.result(timeout=10)
            else:
                vectors = model._model.encode([text], convert_to_numpy=True)
            return np.array(vectors[0], dtype=np.float32)
        except Exception as e:
            logger.warning("SemanticCache: embedding failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Public API (synchronous — called from execute_tool which is sync-safe)
    # ------------------------------------------------------------------

    def get(self, tool_name: str, args: Dict[str, Any]) -> Optional[Any]:
        """Return cached result if a semantically equivalent call exists."""
        if not self._entries:
            return None

        key = _canonical(tool_name, args)
        query_vec = self._embed_sync(key)

        if query_vec is None:
            # Embedding unavailable — fall back to exact match
            return self._exact_get(key)

        best_sim, best_result = 0.0, None
        for stored_vec, stored_result in self._entries:
            sim = _cosine(query_vec, stored_vec)
            if sim > best_sim:
                best_sim, best_result = sim, stored_result

        if best_sim >= self.threshold:
            logger.debug(
                "SemanticCache HIT tool=%s similarity=%.4f threshold=%.4f",
                tool_name, best_sim, self.threshold,
            )
            return best_result

        logger.debug(
            "SemanticCache MISS tool=%s best_similarity=%.4f threshold=%.4f",
            tool_name, best_sim, self.threshold,
        )
        return None

    def set(self, tool_name: str, args: Dict[str, Any], result: Any) -> None:
        """Store a result with its embedding."""
        key = _canonical(tool_name, args)
        vec = self._embed_sync(key)
        if vec is not None:
            self._entries.append((vec, result))
        else:
            # Fallback: store without embedding (exact-match only)
            self._fallback[key] = result
        logger.debug("SemanticCache SET tool=%s entries=%d", tool_name, len(self._entries))

    def clear(self) -> None:
        self._entries.clear()
        self._fallback.clear()

    def __len__(self) -> int:
        return len(self._entries) + len(self._fallback)

    # ------------------------------------------------------------------
    # Exact-match fallback (used when embedding model unavailable)
    # ------------------------------------------------------------------

    @property
    def _fallback(self) -> Dict[str, Any]:
        if not hasattr(self, "_fallback_store"):
            self._fallback_store: Dict[str, Any] = {}
        return self._fallback_store

    def _exact_get(self, key: str) -> Optional[Any]:
        return self._fallback.get(key)


# Module-level singleton — import and use this directly instead of
# instantiating SemanticCache yourself.
semantic_cache = SemanticCache()
