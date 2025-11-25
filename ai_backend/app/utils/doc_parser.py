"""
Document Parser Module

This module provides functionality to parse various document formats into clean plain text.
It is designed to be extensible for future formats like PDF, DOCX, etc.

Usage:
    from app.utils.doc_parser import parse_file, parse_text, RawFormat

    # Parse a file
    text = parse_file("path/to/document.md")

    # Parse raw text
    text = parse_text("# Hello\n\n**World**", format=RawFormat.MARKDOWN)
"""

import os
from enum import Enum
from typing import Optional
import logging

# Third-party imports
import markdown
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class RawFormat(str, Enum):
    """Supported raw data formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    PLAIN = "plain"
    # Future extensions:
    # PDF = "pdf"
    # DOCX = "docx"

def parse_text(data: str, format: RawFormat) -> str:
    """
    Parses raw text data of a given format and returns clean plain text.

    Args:
        data: The raw text data to parse.
        format: The format of the raw data (e.g., RawFormat.MARKDOWN).

    Returns:
        The extracted plain text.

    Raises:
        NotImplementedError: If the format is not yet supported.
    """
    if format == RawFormat.PLAIN:
        return data
    
    elif format == RawFormat.MARKDOWN:
        return _parse_markdown(data)
    
    elif format == RawFormat.HTML:
        return _parse_html(data)
    
    elif format == RawFormat.JSON:
        # TODO: Implement JSON parsing logic (e.g., extract specific fields)
        raise NotImplementedError("JSON parsing is not yet implemented.")
        
    else:
        raise NotImplementedError(f"Parsing for format '{format}' is not yet implemented.")

def parse_file(path: str, format: Optional[RawFormat] = None) -> str:
    """
    Reads a file and parses its content into plain text.
    If format is not provided, it attempts to infer it from the file extension.

    Args:
        path: Path to the file.
        format: Optional explicit format.

    Returns:
        The extracted plain text.
    
    Raises:
        ValueError: If format cannot be inferred.
        FileNotFoundError: If file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    if format is None:
        format = _infer_format_from_extension(path)
        if format is None:
             # Fallback to plain text or raise error? 
             # For now, let's raise an error to be explicit, or default to PLAIN?
             # User requirement said "Start with Markdown", so let's be strict.
             raise ValueError(f"Could not infer format for file: {path}. Please specify format explicitly.")

    with open(path, 'r', encoding='utf-8') as f:
        data = f.read()

    return parse_text(data, format=format)

def _infer_format_from_extension(path: str) -> Optional[RawFormat]:
    """Infers RawFormat from file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.md', '.markdown']:
        return RawFormat.MARKDOWN
    elif ext in ['.html', '.htm']:
        return RawFormat.HTML
    elif ext in ['.json']:
        return RawFormat.JSON
    elif ext in ['.txt', '.text']:
        return RawFormat.PLAIN
    return None

def _parse_markdown(text: str) -> str:
    """
    Converts Markdown to plain text by first converting to HTML 
    and then stripping tags.
    """
    # 1. Convert Markdown to HTML
    # We use 'extra' extension for tables, footnotes, etc.
    html = markdown.markdown(text, extensions=['extra'])
    
    # 2. Extract text from HTML
    return _parse_html(html)

def _parse_html(html_content: str) -> str:
    """
    Extracts text from HTML using BeautifulSoup.
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Add newline after block elements to ensure separation
    # We use a list of common block elements
    block_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div', 'blockquote', 'pre', 'br', 'hr']
    for tag in soup.find_all(block_tags):
        tag.insert_after('\n')
        
    # get_text with empty separator to preserve inline formatting (no extra spaces)
    text = soup.get_text(separator='')
    
    # Clean up excessive whitespace
    lines = [line.strip() for line in text.splitlines()]
    # Remove empty lines
    clean_text = '\n'.join(line for line in lines if line)
    
    return clean_text
