import requests
import json
import time
import os
from dotenv import load_dotenv
from typing import Optional

class FallbackModel:
    """
    Класс для автоматического переключения между OpenRouter и Ollama
    при недоступности одного из API.
    """

    def __init__(
            self,
            openrouter_api_key: Optional[str] = None,
            openrouter_model: str = "openai/gpt-5.4-nano",
            ollama_model: str = "hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_S",
            ollama_url: str = "http://localhost:11434",
            timeout: int = 120,
            retries: int = 2
    ):
        load_dotenv()
        self.openrouter_api_key = openrouter_api_key or os.getenv('OPENROUTER_API_KEY')
        if not self.openrouter_api_key:
            raise ValueError(
                "OpenRouter API key is required. Provide it as argument "
                "or set OPENROUTER_API_KEY in .env file."
            )
        self.openrouter_model = openrouter_model
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.timeout = timeout
        self.retries = retries
        self._primary_available = True

    def _call_openrouter(self, prompt: str) -> Optional[str]:
        """Вызов OpenRouter API (один запрос, без истории)."""
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7
        }

        for attempt in range(self.retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    data=json.dumps(payload),
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']
                else:
                    print(f"OpenRouter error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"OpenRouter attempt {attempt+1} failed: {e}")
                time.sleep(1)
        return None

    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Вызов Ollama API (генерация без истории)."""
        url = f"{self.ollama_url}/api/generate"
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False
        }

        for attempt in range(self.retries):
            try:
                response = requests.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                if response.status_code == 200:
                    return response.json().get('response')
                else:
                    print(f"Ollama error {response.status_code}: {response.text}")
            except Exception as e:
                print(f"Ollama attempt {attempt+1} failed: {e}")
                time.sleep(1)
        return None

    def generate(self, prompt: str) -> Optional[str]:
        """
        Основной метод: пытается вызвать OpenRouter, при недоступности
        автоматически переключается на Ollama.
        """
        if self._primary_available:
            result = self._call_openrouter(prompt)
            if result is not None:
                return result
            else:
                # OpenRouter недоступен – переключаемся на Ollama
                self._primary_available = False
                print("Switching to fallback: Ollama")

        # Используем Ollama
        result = self._call_ollama(prompt)
        if result is not None:
            return result

        # Если и Ollama не работает – возвращаем None
        return None

    def process_query(self, query: str) -> str:
        """
        Метод для совместимости с LLMAgent (как в main.py).
        Возвращает строку с ответом или сообщение об ошибке.
        """
        result = self.generate(query)
        if result is None:
            return "Извините, не удалось получить ответ ни от одного из API."
        return result

    def reset_primary(self):
        """Сброс флага для повторной попытки использования OpenRouter."""
        self._primary_available = True