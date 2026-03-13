import os
from typing import Optional

import httpx


def _env_bool(value, default=True):
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def geocode_city(query: str) -> Optional[dict]:
    if not query:
        return None

    if not _env_bool(os.getenv("GEOCODING_ENABLED"), default=True):
        return None

    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": 1}
    headers = {"User-Agent": "ClimateLens/1.0 (demo)"}

    try:
        response = httpx.get(url, params=params, headers=headers, timeout=4.0)
        response.raise_for_status()
        data = response.json()
    except Exception:
        return None

    if not data:
        return None

    item = data[0]
    try:
        lat = float(item["lat"])
        lon = float(item["lon"])
    except Exception:
        return None

    return {
        "name": item.get("display_name", query.strip()),
        "lat": lat,
        "lon": lon,
        "source": "geocode",
    }
