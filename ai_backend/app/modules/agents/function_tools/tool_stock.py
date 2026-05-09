"""Stock price tool for agent system."""
import yfinance as yf
from typing import Dict, Any


def get_stock_price(symbol: str) -> Dict[str, Any]:
    """Get the latest stock price for a given ticker symbol"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period="1d")
        
        if data.empty:
            return {"symbol": symbol, "error": "No data found for this symbol", "status": "error"}
        
        latest = data.iloc[-1]
        return {
            "symbol": symbol,
            "price": round(latest["Close"], 2),
            "status": "success"
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "status": "error"}