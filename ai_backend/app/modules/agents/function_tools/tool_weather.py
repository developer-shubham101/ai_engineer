"""Weather tool for agent system."""
import requests
from typing import Dict, Any


def get_weather(city: str) -> Dict[str, Any]:
    """Get current weather for a city."""
    try:
        # Using OpenWeatherMap free API (demo key)
        url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid=demo&units=metric"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 401:
            # Demo fallback
            return {
                "city": city,
                "temperature": "22°C",
                "description": "Partly cloudy",
                "humidity": "65%",
                "status": "demo_data"
            }
        
        response.raise_for_status()
        data = response.json()
        
        return {
            "city": data.get("name", city),
            "temperature": f"{data['main']['temp']}°C",
            "description": data["weather"][0]["description"],
            "humidity": f"{data['main']['humidity']}%",
            "status": "success"
        }
    except Exception as e:
        return {"city": city, "error": str(e), "status": "error"}