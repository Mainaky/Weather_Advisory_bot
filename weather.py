"""Small Open-Meteo client: geocode a place and fetch the requested forecast."""
from datetime import datetime
import requests

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CURRENT = "temperature_2m,apparent_temperature,wind_speed_10m,wind_gusts_10m,precipitation,precipitation_probability,uv_index,weather_code,pressure_msl,relative_humidity_2m"
HOURLY = "temperature_2m,apparent_temperature,wind_speed_10m,wind_gusts_10m,precipitation,precipitation_probability,uv_index,weather_code,pressure_msl,relative_humidity_2m"
DAILY = "precipitation_sum,precipitation_probability_max,wind_gusts_10m_max,uv_index_max,weather_code"


def geocode(place: str) -> dict:
    try:
        response = requests.get(GEOCODE_URL, params={"name": place, "count": 1, "language": "en", "format": "json"}, timeout=10)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            raise ValueError(f"No location found for '{place}'.")
        return results[0]
    except requests.RequestException as exc:
        raise RuntimeError("The location service could not be reached.") from exc


def fetch_forecast(location: dict, time_ref: str = "today") -> dict:
    params = {"latitude": location["latitude"], "longitude": location["longitude"], "timezone": "auto",
              "current": CURRENT, "hourly": HOURLY, "daily": DAILY, "wind_speed_unit": "kmh", "forecast_days": 3}
    try:
        response = requests.get(FORECAST_URL, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as exc:
        raise RuntimeError("The weather service could not be reached.") from exc
    try:
        day_index = 1 if time_ref == "tomorrow" else 0
        date = data["daily"]["time"][day_index]
        daily = data["daily"]
        facts = {"place": location.get("name", "your location"), "date": date,
                 "daily_precipitation_sum": daily["precipitation_sum"][day_index],
                 "precipitation_probability_max": daily["precipitation_probability_max"][day_index],
                 "wind_gusts_10m_max": daily["wind_gusts_10m_max"][day_index],
                 "uv_index_max": daily["uv_index_max"][day_index]}
        if time_ref == "now":
            facts.update({key: data["current"][key] for key in ("temperature_2m", "apparent_temperature", "wind_speed_10m", "wind_gusts_10m", "precipitation", "precipitation_probability", "uv_index", "weather_code", "pressure_msl", "relative_humidity_2m")})
            facts["forecast_hour"] = datetime.fromisoformat(data["current"]["time"]).hour
        else:
            hourly = data["hourly"]
            target_hour = datetime.fromisoformat(data["current"]["time"]).hour if time_ref == "today" else 19 if time_ref == "evening" else 12
            target = datetime.fromisoformat(f"{date}T{target_hour:02d}:00")
            candidates = [(abs((datetime.fromisoformat(value) - target).total_seconds()), i) for i, value in enumerate(hourly["time"]) if value.startswith(date)]
            if not candidates:
                raise ValueError("No hourly forecast is available for the requested time.")
            i = min(candidates)[1]
            facts.update({key: hourly[key][i] for key in ("temperature_2m", "apparent_temperature", "wind_speed_10m", "wind_gusts_10m", "precipitation", "precipitation_probability", "uv_index", "weather_code", "pressure_msl", "relative_humidity_2m")})
            facts["forecast_hour"] = datetime.fromisoformat(hourly["time"][i]).hour
        return facts
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise RuntimeError("The weather service returned incomplete forecast data.") from exc
