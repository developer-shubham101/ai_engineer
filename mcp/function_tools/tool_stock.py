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
            "currency": "USD",
            "timestamp": str(latest.name),
            "status": "success"
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "status": "error"}


def get_stock_history(symbol: str, period: str = "5y") -> Dict[str, Any]:
    """Get historical stock prices for a ticker symbol."""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            return {"symbol": symbol, "period": period, "error": "No historical data found", "status": "error"}

        history = []
        for idx, row in data.tail(30).iterrows():
            history.append({
                "date": str(idx.date()),
                "close": round(row["Close"], 2),
                "volume": int(row.get("Volume", 0))
            })

        return {
            "symbol": symbol,
            "period": period,
            "history": history,
            "status": "success"
        }
    except Exception as e:
        return {"symbol": symbol, "period": period, "error": str(e), "status": "error"}


def get_crypto_price(symbol: str) -> Dict[str, Any]:
    """Get the current crypto price for a symbol, e.g. BTC-USD."""
    try:
        crypto = yf.Ticker(symbol)
        data = crypto.history(period="1d")
        if data.empty:
            return {"symbol": symbol, "error": "No data found for this symbol", "status": "error"}

        latest = data.iloc[-1]
        return {
            "symbol": symbol,
            "price": round(latest["Close"], 2),
            "currency": "USD",
            "timestamp": str(latest.name),
            "status": "success"
        }
    except Exception as e:
        return {"symbol": symbol, "error": str(e), "status": "error"}
