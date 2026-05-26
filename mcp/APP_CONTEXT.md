# App Context

## Overview

This project exposes a collection of AI agent tools as an **MCP (Model Context Protocol) server**, allowing any MCP-compatible client (Claude Desktop, custom clients, etc.) to call them over stdio.

---

## Architecture

```
client.py  ──stdio──►  server.py  ──imports──►  function_tools/
```

| File | Role |
|---|---|
| `server.py` | FastMCP server — registers all tools and runs the MCP loop |
| `client.py` | Interactive stdio client — lists tools and calls them via REPL |
| `function_tools/` | Pure Python tool implementations (no MCP dependency) |

---

## Tools

| Module | Functions |
|---|---|
| `tool_weather.py` | `get_weather` |
| `tool_stock.py` | `get_stock_price`, `get_stock_history`, `get_crypto_price` |
| `tool_chart.py` | `generate_stock_chart`, `generate_chart` |
| `tool_file.py` | `save_text_file`, `save_research_report` |
| `tool_web_search.py` | `web_search` |
| `tool_web_scraper.py` | `scrape_url` |
| `tool_travel.py` | `search_flights`, `search_hotels`, `estimate_trip_budget`, `search_places`, `search_restaurants`, `generate_itinerary`, `get_local_transport_info`, `get_distance_between_places`, `generate_trip_summary`, `get_currency_exchange`, `get_geo_distance` |

---

## Environment Variables

Copy `.env.example` to `.env` and fill in your keys. The server loads `.env` automatically on startup via `python-dotenv`.

| Variable | Used by | Required |
|---|---|---|
| `OPENWEATHER_API_KEY` | `tool_weather` | No (falls back to demo data) |
| `SERPAPI_KEY` | `tool_web_search` | No (falls back to DuckDuckGo) |

---

## How It Works

1. `server.py` imports every tool function and registers it with `FastMCP` via `mcp.tool()`.
2. When run (`python server.py`), it starts an MCP stdio loop — tools are callable by any MCP client.
3. `client.py` spawns `server.py` as a subprocess, connects via stdio, and provides an interactive REPL.
