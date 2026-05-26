"""MCP server exposing all function_tools as MCP tools."""
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from mcp.server.fastmcp import FastMCP

from function_tools.tool_weather import get_weather
from function_tools.tool_stock import get_stock_price, get_stock_history, get_crypto_price
from function_tools.tool_chart import generate_stock_chart, generate_chart
from function_tools.tool_file import save_text_file, save_research_report
from function_tools.tool_web_search import web_search
from function_tools.tool_web_scraper import scrape_url
from function_tools.tool_travel import (
    search_flights,
    search_hotels,
    estimate_trip_budget,
    search_places,
    search_restaurants,
    generate_itinerary,
    get_local_transport_info,
    get_distance_between_places,
    generate_trip_summary,
    get_currency_exchange,
    get_geo_distance,
)

mcp = FastMCP("ai-tools-server", port=4154)

# Weather
mcp.tool()(get_weather)

# Stock
mcp.tool()(get_stock_price)
mcp.tool()(get_stock_history)
mcp.tool()(get_crypto_price)

# Charts
mcp.tool()(generate_stock_chart)
mcp.tool()(generate_chart)

# File
mcp.tool()(save_text_file)
mcp.tool()(save_research_report)

# Web
mcp.tool()(web_search)
mcp.tool()(scrape_url)

# Travel
mcp.tool()(search_flights)
mcp.tool()(search_hotels)
mcp.tool()(estimate_trip_budget)
mcp.tool()(search_places)
mcp.tool()(search_restaurants)
mcp.tool()(generate_itinerary)
mcp.tool()(get_local_transport_info)
mcp.tool()(get_distance_between_places)
mcp.tool()(generate_trip_summary)
mcp.tool()(get_currency_exchange)
mcp.tool()(get_geo_distance)

if __name__ == "__main__":
    mcp.run(
        transport="sse"
    )
