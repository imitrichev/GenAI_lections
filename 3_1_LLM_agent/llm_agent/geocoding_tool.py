import requests
from typing import Optional, Dict, Any

class GeocodingTool:
    """Инструмент для прямого и обратного геокодирования через OpenStreetMap Nominatim."""
    
    def __init__(self, user_agent: str = "GenAI_Lab1_Student/1.0"):
        self.base_url = "https://nominatim.openstreetmap.org/search"
        self.reverse_url = "https://nominatim.openstreetmap.org/reverse"
        self.headers = {"User-Agent": user_agent}

    def get_coordinates(self, address: str) -> Optional[Dict[str, float]]:
        params = {"q": address, "format": "json", "limit": 1}
        response = requests.get(self.base_url, params=params, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        if data:
            return {"lat": float(data[0]["lat"]), "lon": float(data[0]["lon"])}
        return None

    def get_address(self, lat: float, lon: float) -> Optional[str]:
        params = {"lat": lat, "lon": lon, "format": "json"}
        response = requests.get(self.reverse_url, params=params, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        return data.get("display_name")

    # === ДОБАВЬТЕ ЭТОТ МЕТОД ДЛЯ ИНТЕГРАЦИИ С АГЕНТОМ ===
    def use(self, query: str) -> str:
        """
        Универсальный метод, который вызывает LLM-агент.
        Ожидает строку вида "coords: Москва" или "address: 55.75, 37.62"
        """
        query = query.strip().lower()
        if query.startswith("coords:") or query.startswith("координаты:"):
            address = query.split(":", 1)[1].strip()
            res = self.get_coordinates(address)
            return str(res) if res else "Место не найдено"
            
        elif query.startswith("address:") or query.startswith("адрес:"):
            parts = query.split(":", 1)[1].strip().split(",")
            if len(parts) == 2:
                try:
                    lat, lon = float(parts[0].strip()), float(parts[1].strip())
                    res = self.get_address(lat, lon)
                    return res if res else "Адрес не найден"
                except ValueError:
                    return "Ошибка формата. Используйте: address: широта, долгота"
                    
        return "Неверный формат. Используйте 'coords: <город>' или 'address: <широта>, <долгота>'"