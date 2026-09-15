import requests
import time
from typing import Optional, Dict

class CurrencyConverterTool:
    """
    Инструмент для конвертации валют с кешем на 1 час.
    """
    
    BASE_URL = "https://api.exchangerate-api.com/v4/latest/"
    
    def __init__(self, api_key: str = "", cache_ttl_seconds: int = 3600):
        self.api_key = api_key
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: Dict[str, Dict] = {}
    
    def _is_cache_valid(self, base_currency: str) -> bool:
        if base_currency not in self._cache:
            return False
        cache_entry = self._cache[base_currency]
        return (time.time() - cache_entry["timestamp"]) < self.cache_ttl_seconds
    
    def _fetch_rates(self, base_currency: str) -> Dict[str, float]:
        if self._is_cache_valid(base_currency):
            return self._cache[base_currency]["rates"]
        
        url = f"{self.BASE_URL}{base_currency}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        
        rates = data.get("rates", {})
        self._cache[base_currency] = {
            "rates": rates,
            "timestamp": time.time()
        }
        return rates
    
    def convert(self, amount: float, from_currency: str, to_currency: str) -> float:
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        rates = self._fetch_rates(from_currency)
        
        if to_currency not in rates:
            raise ValueError(f"Валюта {to_currency} не найдена")
        
        return amount * rates[to_currency]
    
    def get_rate(self, from_currency: str, to_currency: str) -> float:
        from_currency = from_currency.upper()
        to_currency = to_currency.upper()
        
        rates = self._fetch_rates(from_currency)
        
        if to_currency not in rates:
            raise ValueError(f"Валюта {to_currency} не найдена")
        
        return rates[to_currency]
    
    def clear_cache(self):
        self._cache.clear()