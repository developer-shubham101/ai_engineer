"""Travel tools for the smart_travel_planner workflow.

Real APIs used:
  - OpenStreetMap Nominatim  (geocoding, no key needed)
  - exchangerate.host        (currency conversion, no key needed)

All other tools use structured demo/simulated data.
"""
import math
import requests
from typing import Dict, Any, List, Optional


# ---------------------------------------------------------------------------
# Flights
# ---------------------------------------------------------------------------

def search_flights(origin: str, destination: str, date: str = "", budget: str = "") -> Dict[str, Any]:
    """Search for flights between two cities."""
    try:
        return {
            "origin": origin,
            "destination": destination,
            "date": date or "flexible",
            "flights": [
                {"airline": "IndiGo",    "price": "₹3,500", "duration": "2h 10m", "type": "direct"},
                {"airline": "Air India", "price": "₹4,200", "duration": "2h 30m", "type": "direct"},
                {"airline": "SpiceJet",  "price": "₹2,900", "duration": "3h 15m", "type": "1 stop"},
            ],
            "cheapest": "₹2,900",
            "status": "demo_data",
        }
    except Exception as e:
        return {"origin": origin, "destination": destination, "error": str(e), "status": "error"}


# ---------------------------------------------------------------------------
# Hotels
# ---------------------------------------------------------------------------

def search_hotels(destination: str, budget: str = "", days: str = "") -> Dict[str, Any]:
    """Search for hotels at a destination."""
    try:
        return {
            "destination": destination,
            "budget_per_night": budget or "any",
            "hotels": [
                {"name": f"{destination} Grand Hotel",  "price_per_night": "₹2,500", "rating": 4.2, "type": "3-star"},
                {"name": f"Budget Inn {destination}",   "price_per_night": "₹800",   "rating": 3.5, "type": "budget"},
                {"name": f"{destination} Luxury Suites","price_per_night": "₹6,000", "rating": 4.8, "type": "5-star"},
            ],
            "status": "demo_data",
        }
    except Exception as e:
        return {"destination": destination, "error": str(e), "status": "error"}


# ---------------------------------------------------------------------------
# Budget estimation
# ---------------------------------------------------------------------------

def estimate_trip_budget(destination: str, days: str = "3", travelers: str = "1") -> Dict[str, Any]:
    """Estimate total trip budget for a destination."""
    try:
        n_days = int(days) if str(days).isdigit() else 3
        n_travelers = int(travelers) if str(travelers).isdigit() else 1

        per_day = {"hotel": 1500, "food": 800, "local_transport": 400, "activities": 600}
        flight_cost = 3500

        daily_total = sum(per_day.values())
        trip_total = (daily_total * n_days + flight_cost) * n_travelers

        return {
            "destination": destination,
            "days": n_days,
            "travelers": n_travelers,
            "breakdown": {
                "flights": f"₹{flight_cost * n_travelers:,}",
                "hotel":   f"₹{per_day['hotel'] * n_days * n_travelers:,}",
                "food":    f"₹{per_day['food']  * n_days * n_travelers:,}",
                "transport": f"₹{per_day['local_transport'] * n_days * n_travelers:,}",
                "activities": f"₹{per_day['activities'] * n_days * n_travelers:,}",
            },
            "estimated_total": f"₹{trip_total:,}",
            "estimated_total_int": trip_total,
            "status": "demo_data",
        }
    except Exception as e:
        return {"destination": destination, "error": str(e), "status": "error"}


# ---------------------------------------------------------------------------
# Places / attractions
# ---------------------------------------------------------------------------

def search_places(destination: str, category: str = "tourist") -> Dict[str, Any]:
    """Search for tourist attractions or places of interest."""
    try:
        places_db: Dict[str, List[Dict[str, str]]] = {
            "goa": [
                {"name": "Baga Beach",        "type": "beach",      "entry": "free"},
                {"name": "Dudhsagar Falls",    "type": "waterfall",  "entry": "₹400"},
                {"name": "Old Goa Churches",   "type": "heritage",   "entry": "free"},
                {"name": "Anjuna Flea Market", "type": "market",     "entry": "free"},
                {"name": "Fort Aguada",        "type": "fort",       "entry": "₹30"},
            ],
            "jaipur": [
                {"name": "Amber Fort",         "type": "fort",       "entry": "₹200"},
                {"name": "Hawa Mahal",         "type": "palace",     "entry": "₹50"},
                {"name": "City Palace",        "type": "palace",     "entry": "₹130"},
                {"name": "Jantar Mantar",      "type": "heritage",   "entry": "₹50"},
                {"name": "Nahargarh Fort",     "type": "fort",       "entry": "₹50"},
            ],
            "kerala": [
                {"name": "Alleppey Backwaters","type": "nature",     "entry": "₹500 (houseboat)"},
                {"name": "Munnar Tea Gardens", "type": "nature",     "entry": "free"},
                {"name": "Periyar Wildlife",   "type": "wildlife",   "entry": "₹150"},
                {"name": "Kovalam Beach",      "type": "beach",      "entry": "free"},
            ],
            "dubai": [
                {"name": "Burj Khalifa",       "type": "landmark",   "entry": "AED 149"},
                {"name": "Dubai Mall",         "type": "shopping",   "entry": "free"},
                {"name": "Palm Jumeirah",      "type": "beach",      "entry": "free"},
                {"name": "Dubai Marina",       "type": "waterfront", "entry": "free"},
            ],
            "italy": [
                {"name": "Colosseum",          "type": "heritage",   "entry": "€16"},
                {"name": "Vatican Museums",    "type": "museum",     "entry": "€17"},
                {"name": "Venice Canals",      "type": "waterway",   "entry": "free"},
                {"name": "Amalfi Coast",       "type": "beach",      "entry": "free"},
            ],
            "rome": [
                {"name": "Colosseum",          "type": "heritage",   "entry": "€16"},
                {"name": "Vatican Museums",    "type": "museum",     "entry": "€17"},
                {"name": "Trevi Fountain",     "type": "landmark",   "entry": "free"},
                {"name": "Roman Forum",        "type": "heritage",   "entry": "€16"},
            ],
        }
        key = destination.lower()
        attractions = places_db.get(key, [
            {"name": f"{destination} City Center",  "type": "general", "entry": "free"},
            {"name": f"{destination} Museum",       "type": "museum",  "entry": "₹50"},
            {"name": f"{destination} Local Market", "type": "market",  "entry": "free"},
        ])
        return {"destination": destination, "category": category, "places": attractions, "status": "demo_data"}
    except Exception as e:
        return {"destination": destination, "error": str(e), "status": "error"}


# ---------------------------------------------------------------------------
# Restaurants
# ---------------------------------------------------------------------------

def search_restaurants(destination: str, cuisine: str = "local") -> Dict[str, Any]:
    """Search for restaurants at a destination."""
    try:
        return {
            "destination": destination,
            "cuisine": cuisine,
            "restaurants": [
                {"name": f"Spice Garden {destination}", "cuisine": "local",       "avg_cost": "₹400/person", "rating": 4.3},
                {"name": f"The {destination} Kitchen",  "cuisine": "multi-cuisine","avg_cost": "₹600/person", "rating": 4.1},
                {"name": "Street Food Corner",          "cuisine": "street food",  "avg_cost": "₹150/person", "rating": 4.5},
            ],
            "status": "demo_data",
        }
    except Exception as e:
        return {"destination": destination, "error": str(e), "status": "error"}


# ---------------------------------------------------------------------------
# Itinerary generation
# ---------------------------------------------------------------------------

def generate_itinerary(destination: str, days: str = "3", budget: str = "") -> Dict[str, Any]:
    """Generate a day-wise itinerary for a destination."""
    try:
        n_days = int(days) if str(days).isdigit() else 3

        templates = {
            "goa": {
                1: {"morning": "Arrive & check-in hotel", "afternoon": "Baga Beach & water sports", "evening": "Sunset at Calangute Beach", "night": "Seafood dinner at beach shack"},
                2: {"morning": "Old Goa Churches heritage walk", "afternoon": "Dudhsagar Falls excursion", "evening": "Anjuna Flea Market", "night": "Night market at Arpora"},
                3: {"morning": "Fort Aguada sightseeing", "afternoon": "Panjim city walk & local food", "evening": "Depart / return planning", "night": ""},
            },
            "jaipur": {
                1: {"morning": "Arrive & check-in", "afternoon": "Amber Fort & Jaigarh Fort", "evening": "Hawa Mahal photo stop", "night": "Dinner at Chokhi Dhani"},
                2: {"morning": "City Palace & Jantar Mantar", "afternoon": "Johari Bazaar shopping", "evening": "Nahargarh Fort sunset", "night": "Rajasthani folk show"},
                3: {"morning": "Albert Hall Museum", "afternoon": "Local handicraft shopping", "evening": "Depart", "night": ""},
            },
        }

        key = destination.lower()
        day_template = templates.get(key, {})

        itinerary = []
        for day in range(1, n_days + 1):
            plan = day_template.get(day, {
                "morning":   f"Day {day}: Explore {destination} landmarks",
                "afternoon": f"Day {day}: Local cuisine & culture",
                "evening":   f"Day {day}: Leisure & shopping",
                "night":     f"Day {day}: Rest at hotel",
            })
            itinerary.append({"day": day, **plan})

        return {
            "destination": destination,
            "days": n_days,
            "itinerary": itinerary,
            "status": "demo_data",
        }
    except Exception as e:
        return {"destination": destination, "error": str(e), "status": "error"}


# ---------------------------------------------------------------------------
# Local transport
# ---------------------------------------------------------------------------

def get_local_transport_info(destination: str) -> Dict[str, Any]:
    """Get local transport options at a destination."""
    try:
        return {
            "destination": destination,
            "options": [
                {"mode": "Auto-rickshaw", "avg_fare": "₹50–150/trip",  "best_for": "short distances"},
                {"mode": "Taxi / Ola",    "avg_fare": "₹200–500/trip", "best_for": "airport, long trips"},
                {"mode": "Rental bike",   "avg_fare": "₹300–500/day",  "best_for": "self-exploration"},
                {"mode": "Local bus",     "avg_fare": "₹10–30/trip",   "best_for": "budget travel"},
            ],
            "tip": f"Rental bikes are the most popular way to explore {destination} independently.",
            "status": "demo_data",
        }
    except Exception as e:
        return {"destination": destination, "error": str(e), "status": "error"}


# ---------------------------------------------------------------------------
# Distance
# ---------------------------------------------------------------------------

def get_distance_between_places(origin: str, destination: str) -> Dict[str, Any]:
    """Get approximate distance and travel time between two places."""
    try:
        distances = {
            ("delhi", "goa"):     {"km": 1900, "flight": "2h 30m", "train": "~36h"},
            ("delhi", "jaipur"):  {"km": 280,  "flight": "1h",     "train": "4h 30m"},
            ("delhi", "mumbai"):  {"km": 1400, "flight": "2h",     "train": "16h"},
            ("mumbai", "goa"):    {"km": 590,  "flight": "1h 15m", "train": "8h"},
            ("delhi", "kerala"):  {"km": 2900, "flight": "3h",     "train": "~48h"},
        }
        key = (origin.lower(), destination.lower())
        rev_key = (destination.lower(), origin.lower())
        info = distances.get(key) or distances.get(rev_key) or {"km": "N/A", "flight": "varies", "train": "varies"}
        return {"origin": origin, "destination": destination, **info, "status": "demo_data"}
    except Exception as e:
        return {"origin": origin, "destination": destination, "error": str(e), "status": "error"}


# ---------------------------------------------------------------------------
# Trip summary
# ---------------------------------------------------------------------------

def generate_trip_summary(destination: str, days: str = "3", budget: str = "") -> Dict[str, Any]:
    """Generate a concise trip summary with highlights."""
    try:
        return {
            "destination": destination,
            "duration": f"{days} days",
            "best_time_to_visit": "October – March",
            "highlights": [
                f"Explore {destination}'s top attractions",
                "Experience local cuisine and culture",
                "Enjoy nature and outdoor activities",
            ],
            "travel_tips": [
                "Book hotels in advance during peak season",
                "Carry cash for local markets and street food",
                "Respect local customs and dress codes",
                "Stay hydrated and use sunscreen",
            ],
            "estimated_budget": budget or "Varies by preference",
            "status": "demo_data",
        }
    except Exception as e:
        return {"destination": destination, "error": str(e), "status": "error"}


# ---------------------------------------------------------------------------
# Real API: Currency conversion  (exchangerate.host — free, no key)
# ---------------------------------------------------------------------------

# open.er-api.com — free tier, no key required
_CURRENCY_API = "https://open.er-api.com/v6/latest/{base}"

# Common currency aliases users might type
_CURRENCY_ALIASES: Dict[str, str] = {
    "rupee": "INR", "rupees": "INR", "inr": "INR", "rs": "INR", "₹": "INR",
    "yuan": "CNY", "rmb": "CNY", "cny": "CNY",
    "ruble": "RUB", "rubles": "RUB", "rub": "RUB", "₽": "RUB",
    "dollar": "USD", "dollars": "USD", "usd": "USD", "$": "USD",
    "euro": "EUR", "euros": "EUR", "eur": "EUR", "€": "EUR",
    "pound": "GBP", "pounds": "GBP", "gbp": "GBP", "£": "GBP",
    "dirham": "AED", "aed": "AED",
    "baht": "THB", "thb": "THB",
    "yen": "JPY", "jpy": "JPY",
    "won": "KRW", "krw": "KRW",
    "lira": "TRY", "try": "TRY",
    "riyal": "SAR", "sar": "SAR",
    "dinar": "KWD", "kwd": "KWD",
}


def resolve_currency_code(raw: str) -> str:
    """Normalize a currency name/symbol to ISO 4217 code."""
    return _CURRENCY_ALIASES.get(raw.lower().strip(), raw.upper().strip())


def get_currency_exchange(from_currency: str, to_currency: str, amount: float = 1.0) -> Dict[str, Any]:
    """Convert amount between currencies using real exchange rates (open.er-api.com)."""
    try:
        frm = resolve_currency_code(from_currency)
        to = resolve_currency_code(to_currency)
        resp = requests.get(_CURRENCY_API.format(base=frm), timeout=8)
        resp.raise_for_status()
        data = resp.json()
        if data.get("result") != "success":
            raise ValueError(data.get("error-type", "API error"))
        rate = data["rates"].get(to)
        if rate is None:
            raise ValueError(f"Unknown currency: {to}")
        return {
            "from_currency": frm,
            "to_currency": to,
            "original_amount": amount,
            "converted_amount": round(rate * amount, 2),
            "rate": round(rate, 6),
            "status": "success",
        }
    except Exception as e:
        return {"from_currency": from_currency, "to_currency": to_currency,
                "amount": amount, "error": str(e), "status": "error"}


# ---------------------------------------------------------------------------
# Real API: Geo distance via OpenStreetMap Nominatim + Haversine
# ---------------------------------------------------------------------------

_NOMINATIM = "https://nominatim.openstreetmap.org/search"
_NOMINATIM_HEADERS = {"User-Agent": "ai-travel-planner/1.0"}


def _geocode(place: str) -> Optional[tuple]:
    """Return (lat, lon) for a place name using OSM Nominatim."""
    try:
        resp = requests.get(
            _NOMINATIM,
            params={"q": place, "format": "json", "limit": 1},
            headers=_NOMINATIM_HEADERS,
            timeout=8,
        )
        resp.raise_for_status()
        results = resp.json()
        if results:
            return float(results[0]["lat"]), float(results[0]["lon"])
    except Exception:
        pass
    return None


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return round(R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)), 1)


def get_geo_distance(origin: str, destination: str) -> Dict[str, Any]:
    """Get real straight-line distance between two places via OpenStreetMap."""
    try:
        o_coords = _geocode(origin)
        d_coords = _geocode(destination)
        if not o_coords or not d_coords:
            missing = origin if not o_coords else destination
            return {"origin": origin, "destination": destination,
                    "error": f"Could not geocode: {missing}", "status": "error"}
        km = _haversine_km(*o_coords, *d_coords)
        # Rough flight time estimate: avg 800 km/h
        flight_h = round(km / 800, 1)
        return {
            "origin": origin,
            "destination": destination,
            "origin_coords": {"lat": o_coords[0], "lon": o_coords[1]},
            "destination_coords": {"lat": d_coords[0], "lon": d_coords[1]},
            "straight_line_km": km,
            "estimated_flight_hours": flight_h,
            "note": "Straight-line distance via OpenStreetMap Nominatim",
            "status": "success",
        }
    except Exception as e:
        return {"origin": origin, "destination": destination, "error": str(e), "status": "error"}
