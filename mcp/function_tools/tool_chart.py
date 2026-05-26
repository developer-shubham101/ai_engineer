"""Chart generation tools for the agent system."""
import os
from pathlib import Path
from typing import Dict, Any, List


def generate_stock_chart(symbol: str, period: str = "5y") -> Dict[str, Any]:
    """Generate a chart for stock history if matplotlib is available."""
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        return {
            "symbol": symbol,
            "period": period,
            "status": "demo",
            "note": "matplotlib not installed; chart metadata only",
        }

    try:
        import yfinance as yf
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        if data.empty:
            return {"symbol": symbol, "period": period, "status": "error", "error": "No history data"}

        chart_dir = Path("charts")
        chart_dir.mkdir(parents=True, exist_ok=True)
        chart_path = chart_dir / f"{symbol}_{period}.png"

        plt.figure(figsize=(8, 4))
        plt.plot(data.index, data["Close"].astype(float), label="Close")
        plt.title(f"{symbol} price history ({period})")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()

        return {
            "symbol": symbol,
            "period": period,
            "chart_path": str(chart_path.resolve()),
            "status": "success",
        }
    except Exception as e:
        return {"symbol": symbol, "period": period, "status": "error", "error": str(e)}


def generate_chart(title: str, data: Any, chart_type: str = "line") -> Dict[str, Any]:
    """Generate a simple chart from structured data."""
    try:
        from matplotlib import pyplot as plt
    except ImportError:
        return {
            "title": title,
            "chart_type": chart_type,
            "status": "demo",
            "note": "matplotlib not installed; chart metadata only",
        }

    try:
        chart_dir = Path("charts")
        chart_dir.mkdir(parents=True, exist_ok=True)
        chart_path = chart_dir / f"{title.replace(' ', '_')[:40]}_{chart_type}.png"

        x_values: List[Any] = []
        y_values: List[float] = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "x" in item and "y" in item:
                    x_values.append(item["x"])
                    y_values.append(float(item["y"]))
        elif isinstance(data, dict) and "history" in data:
            for item in data["history"]:
                x_values.append(item.get("date"))
                y_values.append(float(item.get("close", 0)))
        else:
            raise ValueError("Unsupported data format for chart generation")

        if not x_values or not y_values:
            raise ValueError("No numeric data provided for chart")

        plt.figure(figsize=(8, 4))
        if chart_type == "bar":
            plt.bar(x_values, y_values)
        else:
            plt.plot(x_values, y_values, marker="o")
        plt.title(title)
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(chart_path)
        plt.close()

        return {
            "title": title,
            "chart_type": chart_type,
            "chart_path": str(chart_path.resolve()),
            "status": "success",
        }
    except Exception as e:
        return {"title": title, "chart_type": chart_type, "status": "error", "error": str(e)}
