import time
from typing import Dict, Tuple

import requests


class WeatherTool:
    """Инструмент для получения текущей погоды по названию города."""

    name = "weather"
    description = (
        "Получает текущую погоду по названию города. "
        "Результат кешируется на 10 минут."
    )

    GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"
    CACHE_TTL_SECONDS = 10 * 60

    WEATHER_CODES = {
        0: "ясно",
        1: "преимущественно ясно",
        2: "переменная облачность",
        3: "пасмурно",
        45: "туман",
        48: "изморозь и туман",
        51: "слабая морось",
        53: "морось",
        55: "сильная морось",
        61: "слабый дождь",
        63: "дождь",
        65: "сильный дождь",
        71: "слабый снег",
        73: "снег",
        75: "сильный снег",
        80: "слабые ливни",
        81: "ливни",
        82: "сильные ливни",
        95: "гроза",
        96: "гроза с градом",
        99: "сильная гроза с градом",
    }

    def __init__(self) -> None:
        # city_key -> (time_saved, ready_text)
        self._cache: Dict[str, Tuple[float, str]] = {}

    def _get_coordinates(self, city: str) -> Tuple[float, float, str]:
        """Возвращает широту, долготу и отображаемое название города."""
        response = requests.get(
            self.GEO_URL,
            params={
                "name": city,
                "count": 1,
                "language": "ru",
                "format": "json",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results", [])
        if not results:
            raise ValueError(f"город '{city}' не найден")

        city_data = results[0]
        return (
            float(city_data["latitude"]),
            float(city_data["longitude"]),
            city_data.get("name", city),
        )

    def _get_current_weather(self, latitude: float, longitude: float) -> dict:
        """Получает текущие погодные параметры для координат."""
        response = requests.get(
            self.WEATHER_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,apparent_temperature,"
                    "relative_humidity_2m,weather_code,wind_speed_10m"
                ),
                "timezone": "auto",
            },
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()

        current = data.get("current")
        if not current:
            raise ValueError("API не вернул текущую погоду")

        return current

    def _fetch_weather(self, city: str) -> str:
        """Получает свежую погоду из API и форматирует ответ."""
        latitude, longitude, display_name = self._get_coordinates(city)
        current = self._get_current_weather(latitude, longitude)

        temperature = current.get("temperature_2m")
        apparent = current.get("apparent_temperature")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        code = current.get("weather_code")
        description = self.WEATHER_CODES.get(code, "неизвестные условия")

        return (
            f"Погода в городе {display_name}: {temperature} °C, {description}; "
            f"ощущается как {apparent} °C; влажность {humidity}%; "
            f"ветер {wind} км/ч."
        )

    def use(self, city: str) -> str:
        """Возвращает текущую погоду, используя кеш в течение 10 минут."""
        city = city.strip()
        if not city:
            return "Ошибка: укажите название города."

        cache_key = city.casefold()
        now = time.monotonic()
        cached = self._cache.get(cache_key)

        if cached is not None:
            saved_at, result = cached
            if now - saved_at < self.CACHE_TTL_SECONDS:
                return result

        try:
            result = self._fetch_weather(city)
        except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
            return f"Ошибка получения погоды для '{city}': {exc}"

        self._cache[cache_key] = (now, result)
        return result
