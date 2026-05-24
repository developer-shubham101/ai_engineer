"""Lazy registry mapping tool names to callable implementations."""
from __future__ import annotations

import logging
from typing import Callable, Dict, Any

_TOOL_BUILDERS: Dict[str, Callable] = {}
_logger = logging.getLogger(__name__)


def get_tool_registry() -> Dict[str, Callable]:
    """Lazily build and return the tool-name → callable map.

    Names are kept in sync with agent_runner.REGISTRY so /tools and
    /tools/{tool_name}/test work for both custom and AutoGen orchestrators.
    """
    if _TOOL_BUILDERS:
        return _TOOL_BUILDERS

    _logger.debug("[ToolRegistry] building tool registry")

    from ...function_tools.tool_web_search import web_search
    from ...function_tools.tool_web_scraper import scrape_url
    from ...function_tools.tool_stock import get_stock_price, get_stock_history, get_crypto_price
    from ...function_tools.tool_weather import get_weather
    from ...function_tools.tool_chart import generate_stock_chart, generate_chart
    from ...function_tools.tool_file import save_research_report
    from ...function_tools.tool_travel import (
        search_flights, search_hotels, estimate_trip_budget, search_places,
        search_restaurants, generate_itinerary, get_local_transport_info,
        get_distance_between_places, generate_trip_summary,
        get_currency_exchange, get_geo_distance,
    )

    def web_search_tool(query: str) -> Dict[str, Any]:
        """Search the internet for real-time information on any topic."""
        return web_search(query, max_results=5)

    def scrape_url_tool(url: str) -> Dict[str, Any]:
        """Fetch and extract full text content from a URL."""
        return scrape_url(url)

    def get_stock_price_tool(symbol: str) -> Dict[str, Any]:
        """Get the current stock price for a ticker symbol (e.g. AAPL, TSLA)."""
        return get_stock_price(symbol)

    def get_stock_history_tool(symbol: str, period: str = "5y") -> Dict[str, Any]:
        """Get historical stock prices for a ticker symbol."""
        return get_stock_history(symbol, period)

    def generate_stock_chart_tool(symbol: str, period: str = "5y") -> Dict[str, Any]:
        """Generate a stock performance chart for a symbol over a period."""
        return generate_stock_chart(symbol, period)

    def get_crypto_price_tool(symbol: str) -> Dict[str, Any]:
        """Get the current crypto price for a symbol (e.g. BTC-USD)."""
        return get_crypto_price(symbol)

    def generate_chart_tool(title: str, data: Any, chart_type: str = "line") -> Dict[str, Any]:
        """Generate a generic chart from structured data."""
        return generate_chart(title, data, chart_type)

    def get_weather_tool(city: str) -> Dict[str, Any]:
        """Get current weather conditions for a city."""
        return get_weather(city)

    def save_research_report_tool(
        title: str,
        query: str,
        summary: str,
        markdown: str,
        metadata: str,
        sources: str,
    ) -> str:
        """Save a structured research report as markdown + JSON sidecar.

        Args:
            title:    Report title (used as filename base).
            query:    Original research query.
            summary:  Executive summary (1-3 sentences).
            markdown: Full report body in markdown format.
            metadata: JSON string of extra metadata (tags, topic, etc.).
            sources:  Newline-separated list of source URLs or citations.
        """
        result = save_research_report(title, query, summary, markdown, metadata, sources)
        if result.get("status") == "success":
            return (
                f"Report saved: '{result['title']}' "
                f"({result['size']} chars, {result['sources_count']} sources) "
                f"at {result['report_path']}"
            )
        return f"Save failed: {result.get('error')}"

    def search_flights_tool(origin: str, destination: str, date: str = "", budget: str = "") -> Dict[str, Any]:
        """Search for flights between two cities."""
        return search_flights(origin, destination, date, budget)

    def search_hotels_tool(destination: str, budget: str = "", days: str = "") -> Dict[str, Any]:
        """Search for hotels at a destination."""
        return search_hotels(destination, budget, days)

    def estimate_trip_budget_tool(destination: str, days: str = "3", travelers: str = "1") -> Dict[str, Any]:
        """Estimate total trip budget including flights, hotels, food, and activities."""
        return estimate_trip_budget(destination, days, travelers)

    def search_places_tool(destination: str, category: str = "tourist") -> Dict[str, Any]:
        """Search for tourist attractions and places of interest at a destination."""
        return search_places(destination, category)

    def search_restaurants_tool(destination: str, cuisine: str = "local") -> Dict[str, Any]:
        """Search for restaurants and dining options at a destination."""
        return search_restaurants(destination, cuisine)

    def generate_itinerary_tool(destination: str, days: str = "3", budget: str = "") -> Dict[str, Any]:
        """Generate a day-wise travel itinerary for a destination."""
        return generate_itinerary(destination, days, budget)

    def get_local_transport_info_tool(destination: str) -> Dict[str, Any]:
        """Get local transport options (auto, taxi, bus, rental) at a destination."""
        return get_local_transport_info(destination)

    def get_distance_between_places_tool(origin: str, destination: str) -> Dict[str, Any]:
        """Get approximate distance and travel time between two places."""
        return get_distance_between_places(origin, destination)

    def generate_trip_summary_tool(destination: str, days: str = "3", budget: str = "") -> Dict[str, Any]:
        """Generate a concise trip summary with highlights and travel tips."""
        return generate_trip_summary(destination, days, budget)

    def get_currency_exchange_tool(from_currency: str, to_currency: str, amount: float = 1.0) -> Dict[str, Any]:
        """Convert amount between currencies using real exchange rates."""
        return get_currency_exchange(from_currency, to_currency, amount)

    def get_geo_distance_tool(origin: str, destination: str) -> Dict[str, Any]:
        """Get real straight-line distance between two places via OpenStreetMap."""
        return get_geo_distance(origin, destination)

    _TOOL_BUILDERS.update({
        "web_search": web_search_tool,
        "scrape_url": scrape_url_tool,
        "get_stock_price": get_stock_price_tool,
        "get_stock_history": get_stock_history_tool,
        "generate_stock_chart": generate_stock_chart_tool,
        "get_crypto_price": get_crypto_price_tool,
        "generate_chart": generate_chart_tool,
        "get_weather": get_weather_tool,
        "save_research_report": save_research_report_tool,
        "search_flights": search_flights_tool,
        "search_hotels": search_hotels_tool,
        "estimate_trip_budget": estimate_trip_budget_tool,
        "search_places": search_places_tool,
        "search_restaurants": search_restaurants_tool,
        "generate_itinerary": generate_itinerary_tool,
        "get_local_transport_info": get_local_transport_info_tool,
        "get_distance_between_places": get_distance_between_places_tool,
        "generate_trip_summary": generate_trip_summary_tool,
        "get_currency_exchange": get_currency_exchange_tool,
        "get_geo_distance": get_geo_distance_tool,
    })
    _logger.debug("[ToolRegistry] registered %d tools: %s", len(_TOOL_BUILDERS), list(_TOOL_BUILDERS))
    return _TOOL_BUILDERS
