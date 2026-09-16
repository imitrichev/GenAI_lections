import requests
from typing import List, Dict, Optional

class FallbackModel:

    def __init__(
        self,
        openrouter_api_key: Optional[str] = None,
        openrouter_model: str = "tngtech/deepseek-r1t2-chimera",
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen2.5:0.5b"
    ):
        self.openrouter_api_key = openrouter_api_key
        self.openrouter_model = openrouter_model
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        self.openrouter_url = "https://openrouter.ai/api/v1/chat/completions"
        self.ollama_url = f"{self.ollama_base_url}/v1/chat/completions"

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> Dict:
        """
        Пытается сделать запрос к OpenRouter. Если не получается, 
        переключается на локальный Ollama.
        """
        # Попытка использовать OpenRouter
        if self.openrouter_api_key:
            try:
                headers = {
                    "Authorization": f"Bearer {self.openrouter_api_key}",
                    "Content-Type": "application/json"
                }
                payload = {"model": self.openrouter_model, "messages": messages, **kwargs}
                response = requests.post(self.openrouter_url, json=payload, headers=headers, timeout=10)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"[FallbackModel] OpenRouter недоступен ({e}). Переключаюсь на Ollama...")
        
        # Фоллбэк на Ollama
        try:
            headers = {"Content-Type": "application/json"}
            payload = {"model": self.ollama_model, "messages": messages, "stream": False, **kwargs}
            response = requests.post(self.ollama_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"[FallbackModel] Ошибка: оба API (OpenRouter и Ollama) недоступны. Детали: {e}")
