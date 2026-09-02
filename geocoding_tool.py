import requests
from typing import Optional, Dict, Any

class GeocodingTool:
    """Инструмент для прямого и обратного геокодирования через OpenStreetMap Nominatim."""
    
    def __init__(self, user_agent: str = "GenAI_Lab1_Student/1.0"):
        self.base_url = "https://nominatim.openstreetmap.org/search"
        self.reverse_url = "https://nominatim.openstreetmap.org/reverse"
        self.headers = {"User-Agent": user_agent}

    def get_coordinates(self, address: str) -> Optional[Dict[str, float]]:
        """
        Преобразует название места в координаты.
        :param address: Название места (например, "Красная площадь, Москва")
        :return: Словарь с ключами 'lat' и 'lon', или None при ошибке.
        """
        params = {"q": address, "format": "json", "limit": 1}
        response = requests.get(self.base_url, params=params, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        
        if data:
            return {
                "lat": float(data[0]["lat"]),
                "lon": float(data[0]["lon"])
            }
        return None

    def get_address(self, lat: float, lon: float) -> Optional[str]:
        """
        Преобразует координаты в название места.
        :param lat: Широта
        :param lon: Долгота
        :return: Строка с адресом, или None при ошибке.
        """
        params = {"lat": lat, "lon": lon, "format": "json"}
        response = requests.get(self.reverse_url, params=params, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        
        if "display_name" in data:
            return data["display_name"]
        return None