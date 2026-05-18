"""Paragraph-aware chunking for improved retrieval quality.

This module implements semantic chunking that respects paragraph boundaries,
keeping semantic units together for better retrieval quality.
"""
import logging
import re
from typing import List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ChunkConfig:
    """Configuration for paragraph-aware chunking."""
    chunk_size: int = 512  # Target chunk size in characters
    min_chunk_size: int = 100  # Minimum chunk size
    max_chunk_size: int = 1024  # Maximum chunk size before forced split
    overlap: int = 50  # Overlap between chunks


def estimate_tokens(text: str) -> int:
    """Estimate token count (rough approximation: 1 token ≈ 4 characters)."""
    return len(text) // 4


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs on double newlines.
    
    Args:
        text: Input text
        
    Returns:
        List of paragraphs (non-empty)
    """
    # Split on double newlines (paragraph boundaries)
    paragraphs = re.split(r'\n\n+', text)
    
    # Filter out empty paragraphs and strip whitespace
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    return paragraphs


def split_long_paragraph(paragraph: str, max_size: int, overlap: int = 50) -> List[str]:
    """Split a long paragraph into smaller chunks at sentence boundaries.
    
    Args:
        paragraph: Paragraph text
        max_size: Maximum chunk size
        overlap: Overlap between chunks
        
    Returns:
        List of chunks
    """
    # If paragraph fits, return as-is
    if len(paragraph) <= max_size:
        return [paragraph]
    
    # Try to split on sentence boundaries first
    sentences = re.split(r'(?<=[.!?])\s+', paragraph)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_size = len(sentence)
        
        # If single sentence exceeds max_size, split it further
        if sentence_size > max_size:
            # Save current chunk if exists
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Split long sentence on commas or spaces
            if ',' in sentence:
                parts = sentence.split(',')
                for part in parts:
                    if len(part.strip()) > 0:
                        chunks.append(part.strip())
            else:
                # Last resort: split on spaces
                words = sentence.split()
                temp_chunk = []
                temp_size = 0
                for word in words:
                    word_size = len(word) + 1  # +1 for space
                    if temp_size + word_size > max_size and temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                        temp_chunk = []
                        temp_size = 0
                    temp_chunk.append(word)
                    temp_size += word_size
                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
            continue
        
        # Check if adding this sentence would exceed max_size
        if current_size + sentence_size + 1 > max_size and current_chunk:
            # Save current chunk
            chunks.append(' '.join(current_chunk))
            
            # Start new chunk with overlap
            if overlap > 0 and current_chunk:
                # Take last few sentences for overlap
                overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) >= 2 else current_chunk[-1]
                if len(overlap_text) <= overlap:
                    current_chunk = current_chunk[-2:] if len(current_chunk) >= 2 else [current_chunk[-1]]
                    current_size = len(' '.join(current_chunk))
                else:
                    current_chunk = []
                    current_size = 0
            else:
                current_chunk = []
                current_size = 0
        
        current_chunk.append(sentence)
        current_size += sentence_size + 1  # +1 for space
    
    # Add final chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks


def chunk_text_paragraph_aware(
    text: str,
    config: ChunkConfig = None
) -> List[Tuple[str, dict]]:
    """Chunk text with paragraph awareness.
    
    Pipeline:
    1. Split on double newlines (paragraph boundaries)
    2. Combine small paragraphs until reaching target size
    3. Split large paragraphs at sentence boundaries
    4. Add overlap between chunks
    
    Args:
        text: Input text
        config: Chunking configuration
        
    Returns:
        List of (chunk_text, metadata) tuples
    """
    if config is None:
        config = ChunkConfig()
    
    if not text or not text.strip():
        return []
    
    # Step 1: Split into paragraphs
    paragraphs = split_into_paragraphs(text)
    
    if not paragraphs:
        return []
    
    logger.info(f"Split text into {len(paragraphs)} paragraphs")
    
    # Step 2: Combine or split paragraphs to create optimal chunks
    chunks_with_meta = []
    current_chunk_paras = []
    current_size = 0
    
    for i, para in enumerate(paragraphs):
        para_size = len(para)
        
        # If paragraph alone exceeds max_size, split it
        if para_size > config.max_chunk_size:
            # Save current chunk if exists
            if current_chunk_paras:
                chunk_text = '\n\n'.join(current_chunk_paras)
                chunks_with_meta.append((
                    chunk_text,
                    {
                        "chunk_type": "paragraph_aware",
                        "paragraph_count": len(current_chunk_paras),
                        "char_count": len(chunk_text),
                        "token_estimate": estimate_tokens(chunk_text)
                    }
                ))
                current_chunk_paras = []
                current_size = 0
            
            # Split long paragraph
            sub_chunks = split_long_paragraph(para, config.max_chunk_size, config.overlap)
            for sub_chunk in sub_chunks:
                chunks_with_meta.append((
                    sub_chunk,
                    {
                        "chunk_type": "paragraph_aware_split",
                        "paragraph_count": 1,
                        "char_count": len(sub_chunk),
                        "token_estimate": estimate_tokens(sub_chunk),
                        "split_from_long_paragraph": True
                    }
                ))
            continue
        
        # Check if adding this paragraph would exceed target size
        if current_size + para_size + 2 > config.chunk_size and current_chunk_paras:  # +2 for \n\n
            # Save current chunk
            chunk_text = '\n\n'.join(current_chunk_paras)
            chunks_with_meta.append((
                chunk_text,
                {
                    "chunk_type": "paragraph_aware",
                    "paragraph_count": len(current_chunk_paras),
                    "char_count": len(chunk_text),
                    "token_estimate": estimate_tokens(chunk_text)
                }
            ))
            
            # Start new chunk with optional overlap
            if config.overlap > 0 and current_chunk_paras:
                # Include last paragraph for overlap if it's small enough
                last_para = current_chunk_paras[-1]
                if len(last_para) <= config.overlap:
                    current_chunk_paras = [last_para]
                    current_size = len(last_para)
                else:
                    current_chunk_paras = []
                    current_size = 0
            else:
                current_chunk_paras = []
                current_size = 0
        
        # Add paragraph to current chunk
        current_chunk_paras.append(para)
        current_size += para_size + 2  # +2 for \n\n separator
    
    # Add final chunk
    if current_chunk_paras:
        chunk_text = '\n\n'.join(current_chunk_paras)
        # Only add if it meets minimum size
        if len(chunk_text) >= config.min_chunk_size:
            chunks_with_meta.append((
                chunk_text,
                {
                    "chunk_type": "paragraph_aware",
                    "paragraph_count": len(current_chunk_paras),
                    "char_count": len(chunk_text),
                    "token_estimate": estimate_tokens(chunk_text)
                }
            ))
    
    logger.info(f"Created {len(chunks_with_meta)} paragraph-aware chunks")
    
    return chunks_with_meta


def chunk_text_paragraph_aware_simple(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50
) -> List[str]:
    """Simplified interface for paragraph-aware chunking.
    
    Args:
        text: Input text
        chunk_size: Target chunk size
        overlap: Overlap between chunks
        
    Returns:
        List of chunk texts (without metadata)
    """
    config = ChunkConfig(
        chunk_size=chunk_size,
        overlap=overlap,
        max_chunk_size=chunk_size * 2  # Allow 2x for paragraph integrity
    )
    
    chunks_with_meta = chunk_text_paragraph_aware(text, config)
    return [chunk for chunk, _ in chunks_with_meta]


# Backward compatibility
def chunk_text_basic_enhanced(text: str, chunk_size: int = 512, overlap: int = 64) -> List[str]:
    """Enhanced chunking with paragraph awareness (backward compatible interface).
    
    This replaces the old fixed-size chunking with paragraph-aware chunking
    while maintaining the same function signature.
    """
    return chunk_text_paragraph_aware_simple(text, chunk_size, overlap)
