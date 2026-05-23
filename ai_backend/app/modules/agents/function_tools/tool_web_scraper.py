"""Web scraper tool for agent system.

Fetches a URL and extracts clean readable text content.
Used to get full article/page content after a web search.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Max characters to return to avoid flooding the LLM context
MAX_CONTENT_LENGTH = 3000


def scrape_url(url: str) -> Dict[str, Any]:
    """
    Fetch a URL and extract clean readable text content.

    Args:
        url: The URL to scrape

    Returns:
        Dict with extracted text and metadata
    """
    try:
        import requests
        from bs4 import BeautifulSoup

        headers = {"User-Agent": "Mozilla/5.0 (compatible; RAGBot/1.0)"}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
            tag.decompose()

        # Extract main content - prefer article/main tags
        main = soup.find("article") or soup.find("main") or soup.find("body")
        text = main.get_text(separator="\n", strip=True) if main else soup.get_text(separator="\n", strip=True)

        # Clean up excessive blank lines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        clean_text = "\n".join(lines)

        # Truncate to avoid context overflow
        truncated = len(clean_text) > MAX_CONTENT_LENGTH
        content = clean_text[:MAX_CONTENT_LENGTH] + ("..." if truncated else "")

        return {
            "url": url,
            "content": content,
            "char_count": len(clean_text),
            "truncated": truncated,
            "status": "success"
        }

    except ImportError as e:
        return {"url": url, "error": f"Missing dependency: {e}. Run: pip install requests beautifulsoup4", "status": "missing_dependency"}
    except Exception as e:
        logger.error(f"Scrape failed for {url}: {e}")
        return {"url": url, "error": str(e), "status": "error"}
