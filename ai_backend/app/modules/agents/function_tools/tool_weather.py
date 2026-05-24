"""Weather tool — real OpenWeatherMap API with demo fallback."""
import os
import requests
from typing import Dict, Any

_OWM_BASE = "http://api.openweathermap.org/data/2.5/weather"
_OWM_KEY = os.getenv("OPENWEATHER_API_KEY", "")


def get_weather(city: str) -> Dict[str, Any]:
    """Get current weather for a city via OpenWeatherMap."""
    try:
        if not _OWM_KEY:
            return _demo_weather(city, reason="no_api_key")

        resp = requests.get(
            _OWM_BASE,
            params={"q": city, "appid": _OWM_KEY, "units": "metric"},
            timeout=10,
        )
        if resp.status_code == 401:
            return _demo_weather(city, reason="invalid_api_key")
        if resp.status_code == 404:
            return {"city": city, "error": "City not found", "status": "error"}
        resp.raise_for_status()
        d = resp.json()
        return {
            "city": d.get("name", city),
            "country": d.get("sys", {}).get("country", ""),
            "temperature_c": round(d["main"]["temp"], 1),
            "feels_like_c": round(d["main"]["feels_like"], 1),
            "description": d["weather"][0]["description"].capitalize(),
            "humidity_pct": d["main"]["humidity"],
            "wind_kmh": round(d.get("wind", {}).get("speed", 0) * 3.6, 1),
            "status": "success",
        }
    except Exception as e:
        return {"city": city, "error": str(e), "status": "error"}


def _demo_weather(city: str, reason: str = "demo") -> Dict[str, Any]:
    _defaults: Dict[str, Dict[str, Any]] = {
        "goa":       {"temperature_c": 30, "description": "Sunny", "humidity_pct": 75},
        "jaipur":    {"temperature_c": 28, "description": "Clear sky", "humidity_pct": 40},
        "manali":    {"temperature_c": 8,  "description": "Cold and cloudy", "humidity_pct": 60},
        "dubai":     {"temperature_c": 38, "description": "Hot and sunny", "humidity_pct": 50},
        "paris":     {"temperature_c": 15, "description": "Partly cloudy", "humidity_pct": 65},
        "rome":      {"temperature_c": 18, "description": "Mild and sunny", "humidity_pct": 55},
        "italy":     {"temperature_c": 18, "description": "Mild and sunny", "humidity_pct": 55},
        "moscow":    {"temperature_c": -2, "description": "Snow", "humidity_pct": 80},
        "new york":  {"temperature_c": 12, "description": "Partly cloudy", "humidity_pct": 60},
    }
    base = _defaults.get(city.lower(), {"temperature_c": 22, "description": "Partly cloudy", "humidity_pct": 65})
    return {"city": city, "status": f"demo_data:{reason}", **base}
