"""Web search tool for agent system.

Uses DuckDuckGo (no API key required) as primary source.
Falls back to SerpAPI if SERPAPI_KEY is set in environment.
"""
import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def _search_duckduckgo(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search using DuckDuckGo via duckduckgo_search library."""
    try:
        from duckduckgo_search import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", "")
                })
        return results
    except ImportError:
        raise ImportError("duckduckgo_search not installed. Run: pip install duckduckgo-search")
    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        raise


def _search_serpapi(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search using SerpAPI (requires SERPAPI_KEY env var)."""
    try:
        import requests
        api_key = os.getenv("SERPAPI_KEY")
        if not api_key:
            raise ValueError("SERPAPI_KEY not set")

        resp = requests.get(
            "https://serpapi.com/search",
            params={"q": query, "api_key": api_key, "num": max_results},
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        for item in data.get("organic_results", [])[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", "")
            })
        return results
    except Exception as e:
        logger.error(f"SerpAPI search failed: {e}")
        raise


def web_search(query: str, max_results: int = 5) -> Dict[str, Any]:
    """
    Search the internet for real-time information.

    Uses SerpAPI if SERPAPI_KEY is set, otherwise DuckDuckGo (free, no key needed).

    Args:
        query: Search query string
        max_results: Number of results to return (default 5)

    Returns:
        Dict with results list and metadata
    """
    try:
        # Prefer SerpAPI if key is available
        if os.getenv("SERPAPI_KEY"):
            results = _search_serpapi(query, max_results)
            source = "serpapi"
        else:
            results = _search_duckduckgo(query, max_results)
            source = "duckduckgo"

        if not results:
            return {
                "query": query,
                "results": [],
                "status": "no_results",
                "source": source
            }

        # Format results as readable text for LLM
        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(
                f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}"
            )

        return {
            "query": query,
            "results": results,
            "formatted": "\n\n".join(formatted),
            "count": len(results),
            "source": source,
            "status": "success"
        }

    except ImportError as e:
        return {"query": query, "error": str(e), "status": "missing_dependency"}
    except Exception as e:
        logger.error(f"Web search failed for query '{query}': {e}")
        return {"query": query, "error": str(e), "status": "error"}
