import requests
import pytest

def test_ollama_api_is_running():
    """Проверяет, что сервер Ollama запущен и доступен."""
    response = requests.get("http://localhost:11434/api/tags")
    assert response.status_code == 200, "Ollama API не отвечает на порту 11434"

def test_ollama_can_generate_response():
    """Проверяет, что Ollama может сгенерировать ответ на простой запрос."""
    payload = {
        "model": "qwen2.5:0.5b",  # Используем маленькую модель для скорости в CI
        "prompt": "Скажи только слово: ТЕСТ",
        "stream": False
    }
    response = requests.post("http://localhost:11434/api/generate", json=payload)
    
    assert response.status_code == 200, "Запрос к Ollama завершился с ошибкой"
    
    response_data = response.json()
    assert "response" in response_data, "В ответе Ollama нет поля 'response'"
    assert "тест" in response_data["response"].lower() or "test" in response_data["response"].lower(), \
        f"Ollama вернул неожиданный ответ: {response_data['response']}"
