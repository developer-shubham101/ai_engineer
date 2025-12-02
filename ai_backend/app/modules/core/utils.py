"""Core utility functions for the modular architecture."""
from typing import List, Dict, Any, Optional

def chunk_text_basic(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Produce overlapping chunks of the input text.
    Fixed so we always make progress and produce expected overlaps.
    """
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    L = len(text)
    while start < L:
        end = min(start + chunk_size, L)
        chunks.append(text[start:end])
        if end == L:
            break
        # advance start keeping overlap, but ensure progress by at least 1
        start = max(end - overlap, start + 1)
    return chunks


def sanitize_meta_value(val):
    """
    Ensure metadata values are primitives (str, int, float, bool) for Chroma.
    - If val is list of primitives -> join with commas
    - If val is dict -> json.dumps
    - Else convert to str
    """
    import json
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, list):
        # if list of primitives, join; otherwise json-dump
        if all(isinstance(x, (str, int, float, bool)) for x in val):
            return ",".join(str(x) for x in val)
        return json.dumps(val, ensure_ascii=False)
    if isinstance(val, dict):
        return json.dumps(val, ensure_ascii=False)
    # fallback
    return str(val)


def sanitize_metadata_dict(meta: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Sanitize a metadata dictionary for Chroma compatibility."""
    if not meta:
        return {}
    return {str(k): sanitize_meta_value(v) for k, v in meta.items()}


def is_empty(data):
    # Case 1: None
    if data is None:
        return True

    # Case 2: Iterable types (list, dict, tuple, set, string)
    if isinstance(data, (list, dict, tuple, set, str)):
        return len(data) == 0

    # Case 3: Has length
    if hasattr(data, "__len__"):
        return len(data) == 0

    return True


def is_collection_empty(data):
    """Return True if Chroma/Vector DB response represents an empty collection."""
    if data is None:
        return True

    # If dictionary structure
    if isinstance(data, dict):
        # Chroma empty pattern: ids == [] and documents == []
        ids = data.get("ids", [])
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])

        # "Empty" means: all primary fields contain no usable data
        if len(ids) == 0 and len(docs) == 0 and len(metas) == 0:
            return True

        return False  # some data exists

    # If some vector store object
    if hasattr(data, "ids"):
        ids = getattr(data, "ids", [])
        return len(ids) == 0

    if hasattr(data, "documents"):
        docs = getattr(data, "documents", [])
        return len(docs) == 0

    # Generic fallback
    try:
        return len(data) == 0
    except Exception:
        return not bool(data)
