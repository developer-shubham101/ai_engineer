# MCP AI Tools Server

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server that exposes weather, stock, chart, file, web search, web scraping, and travel tools — usable by any MCP-compatible client.

## Setup

```bash
pip install -r requirements.txt
```

## Run the server

```bash
python server.py
```

The server communicates over **stdio** (standard MCP transport).

## Run the interactive client

```bash
python client.py
```

This spawns the server and opens a REPL:

```
> get_weather {"city": "London"}
> get_stock_price {"symbol": "AAPL"}
> web_search {"query": "latest AI news"}
> list
> quit
```

## MCP Inspector

**Option 1 — `mcp dev` (recommended, zero config):**
```bash
mcp dev server.py
```
This launches the server and opens the MCP Inspector UI automatically in your browser.

**Option 2 — SSE mode (manual):**
```bash
python server.py --inspect
# or on a custom port:
python server.py --inspect --port 9000
```
Then open the [MCP Inspector](https://github.com/modelcontextprotocol/inspector) and connect to `http://127.0.0.1:9000/sse`.

---

## Connect from Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "ai-tools": {
      "command": "python",
      "args": ["/absolute/path/to/server.py"]
    }
  }
}
```

## Environment Variables (optional)

Copy `.env.example` to `.env` and fill in your keys — the server loads it automatically:

```bash
cp .env.example .env
```

| Variable | Effect |
|---|---|
| `OPENWEATHER_API_KEY` | Real weather data (falls back to demo without it) |
| `SERPAPI_KEY` | SerpAPI for web search (falls back to DuckDuckGo without it) |

## Tools

| Category | Tools |
|---|---|
| Weather | `get_weather` |
| Stock | `get_stock_price`, `get_stock_history`, `get_crypto_price` |
| Charts | `generate_stock_chart`, `generate_chart` |
| Files | `save_text_file`, `save_research_report` |
| Web | `web_search`, `scrape_url` |
| Travel | `search_flights`, `search_hotels`, `estimate_trip_budget`, `search_places`, `search_restaurants`, `generate_itinerary`, `get_local_transport_info`, `get_distance_between_places`, `generate_trip_summary`, `get_currency_exchange`, `get_geo_distance` |
