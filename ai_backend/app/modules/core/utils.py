"""Core utility functions for the modular architecture."""
from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_by_sections(text: str) -> Optional[List[Dict[str, str]]]:
    """
    Attempt semantic chunking by detecting sections in text.
    Returns list of {"section": str, "text": str} or None if no clear sections found.
    """
    import re
    
    # Look for markdown-style headers or numbered sections
    header_patterns = [
        r'^#{1,6}\s+(.+)$',  # Markdown headers
        r'^\d+\.\s+(.+)$',   # Numbered sections
        r'^[A-Z][A-Z\s]+:?$', # ALL CAPS headers
        r'^\*\*(.+)\*\*$',    # Bold headers
    ]
    
    sections = []
    current_section = "Introduction"
    current_text = []
    
    lines = text.split('\n')
    found_headers = False
    
    for line in lines:
        line = line.strip()
        if not line:
            current_text.append('')
            continue
            
        # Check if line matches any header pattern
        is_header = False
        for pattern in header_patterns:
            match = re.match(pattern, line, re.MULTILINE)
            if match:
                # Save previous section if it has content
                if current_text and any(t.strip() for t in current_text):
                    sections.append({
                        "section": current_section,
                        "text": '\n'.join(current_text).strip()
                    })
                    found_headers = True
                
                # Start new section
                current_section = match.group(1) if match.groups() else line
                current_text = []
                is_header = True
                break
        
        if not is_header:
            current_text.append(line)
    
    # Add final section
    if current_text and any(t.strip() for t in current_text):
        sections.append({
            "section": current_section,
            "text": '\n'.join(current_text).strip()
        })
    
    # Return None if no clear sections found or only one section
    if not found_headers or len(sections) <= 1:
        return None
        
    return sections


def chunk_text_basic(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """
    Produce overlapping chunks with paragraph awareness.
    
    This implementation respects paragraph boundaries (double newlines) and only
    splits paragraphs if they exceed the chunk size limit. This keeps semantic
    units together for better retrieval quality.
    
    Args:
        text: The input text to chunk
        chunk_size: Target size of each chunk (default: 512)
        overlap: Number of characters to overlap between chunks (default: 64)
    
    Returns:
        List of text chunks
    """
    if not text:
        return []
    
    # Use paragraph-aware chunking for better semantic coherence
    from app.modules.core.chunking import chunk_text_paragraph_aware_simple
    return chunk_text_paragraph_aware_simple(text, chunk_size, overlap)

def deprecated_chunk_text_basic(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
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
