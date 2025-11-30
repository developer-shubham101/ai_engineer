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
from pathlib import Path

# Third-party imports
import markdown
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class RawFormat(str, Enum):
    """Supported raw data formats."""
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    CSV = "csv"
    PLAIN = "plain"
    # TODO: Future extensions:
    # PDF = "pdf"
    # DOCX = "docx"
    # XML = "xml"

def parse_text(data: str, data_format: RawFormat) -> str:
    """
    Parses raw text data of a given format and returns clean plain text.

    Args:
        data: The raw text data to parse.
        data_format: The format of the raw data (e.g., RawFormat.MARKDOWN).

    Returns:
        The extracted plain text.

    Raises:
        NotImplementedError: If the format is not yet supported.
    """
    if data_format == RawFormat.PLAIN:
        return data
    
    elif data_format == RawFormat.MARKDOWN:
        return _parse_markdown(data)
    
    elif data_format == RawFormat.HTML:
        return _parse_html(data)
    
    elif data_format == RawFormat.CSV:
        # TODO: Implement CSV parsing logic
        raise NotImplementedError("CSV parsing is not yet implemented.")
    
    elif data_format == RawFormat.JSON:
        # TODO: Implement JSON parsing logic (e.g., extract specific fields)
        raise NotImplementedError("JSON parsing is not yet implemented.")
        
    else:
        raise NotImplementedError(f"Parsing for format '{data_format}' is not yet implemented.")

def parse_file(path: str, data_format: Optional[RawFormat] = None) -> str:
    """
    Reads a file and parses its content into plain text.
    If format is not provided, it attempts to infer it from the file extension.

    Args:
        path: Path to the file.
        data_format: Optional explicit format.

    Returns:
        The extracted plain text.
    
    Raises:
        ValueError: If format cannot be inferred.
        FileNotFoundError: If file does not exist.
    """
    # Validate path to prevent directory traversal
    safe_path = os.path.abspath(path)
    if not os.path.exists(safe_path):
        raise FileNotFoundError(f"File not found: {safe_path}")

    if data_format is None:
        data_format = _infer_format_from_extension(safe_path)
        if data_format is None:
             raise ValueError(f"Could not infer format for file: {safe_path}. Please specify format explicitly.")

    try:
        with open(safe_path, 'r', encoding='utf-8') as f:
            data = f.read()
    except (UnicodeDecodeError, PermissionError) as e:
        raise ValueError(f"Error reading file {safe_path}: {e}")

    return parse_text(data, data_format=data_format)

def _infer_format_from_extension(path: str) -> Optional[RawFormat]:
    """Infers RawFormat from file extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.md', '.markdown']:
        return RawFormat.MARKDOWN
    elif ext in ['.html', '.htm']:
        return RawFormat.HTML
    elif ext in ['.csv']:
        return RawFormat.CSV
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
    block_tags = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div', 'blockquote', 'pre']
    tags_to_modify = soup.find_all(block_tags)
    for tag in tags_to_modify:
        tag.insert_after('\n')
        
    # get_text with empty separator to preserve inline formatting
    text = soup.get_text(separator='')
    
    # Clean up excessive whitespace
    lines = [line.strip() for line in text.splitlines()]
    clean_text = '\n'.join(line for line in lines if line)
    
    return clean_text

def process_raw_data_directory(raw_data_dir: str = "raw_data", output_dir: str = "data") -> None:
    """
    Process all supported files in raw_data directory and save parsed text to data directory.
    
    Args:
        raw_data_dir: Source directory containing raw files
        output_dir: Target directory for parsed .txt files
    """
    raw_path = Path(raw_data_dir)
    output_path = Path(output_dir)
    
    if not raw_path.exists():
        logger.error(f"Raw data directory not found: {raw_path}")
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(exist_ok=True)
    
    # Supported extensions for processing
    supported_extensions = {'.md', '.markdown', '.html', '.htm'}
    # TODO: Add support for .csv, .json, .pdf, .docx
    
    processed_count = 0
    
    for file_path in raw_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                # Parse the file
                parsed_text = parse_file(str(file_path))
                
                # Create output filename with .txt extension
                output_filename = file_path.stem + '.txt'
                output_file_path = output_path / output_filename
                
                # Write parsed text to output file
                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write(parsed_text)
                
                logger.info(f"Processed: {file_path} -> {output_file_path}")
                processed_count += 1
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Processing complete. {processed_count} files processed.")

if __name__ == "__main__":
    # Example usage
    process_raw_data_directory()
