import pytest
import requests
from unittest.mock import patch, MagicMock
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../llm_agent')))
from fallback_model import FallbackModel

def _test_openrouter_success(model, messages):
    """Тест 1: Успешный запрос к OpenRouter (Ollama не вызывается)"""
    with patch('fallback_model.requests.post') as mock_post:
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Hi from OR"}}]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = model.generate(messages)
        assert result["choices"][0]["message"]["content"] == "Hi from OR"
        assert mock_post.call_count == 1
        assert "openrouter.ai" in mock_post.call_args[0][0]

def _test_openrouter_fails_ollama_success(model, messages):
    """Тест 2: Сбой OpenRouter, успешный фоллбэк на Ollama"""
    with patch('fallback_model.requests.post') as mock_post:
        mock_response_ollama = MagicMock()
        mock_response_ollama.json.return_value = {"choices": [{"message": {"content": "Hi from Ollama"}}]}
        mock_response_ollama.raise_for_status = MagicMock()
        
        # Первый вызов падает, второй успешен
        mock_post.side_effect = [
            requests.exceptions.Timeout("OpenRouter timeout"),
            mock_response_ollama
        ]

        result = model.generate(messages)
        assert result["choices"][0]["message"]["content"] == "Hi from Ollama"
        assert mock_post.call_count == 2

def _test_both_fail(model, messages):
    """Тест 3: Сбой обоих API, должно быть выброшено исключение"""
    with patch('fallback_model.requests.post') as mock_post:
        mock_post.side_effect = requests.exceptions.ConnectionError("Network error")

        with pytest.raises(Exception) as exc_info:
            model.generate(messages)
        
        assert "оба API" in str(exc_info.value)

def test_fallback_model_functions():
    """
    Отдельная тестовая функция, которая вызывает 3 юнит-теста 
    для проверки функций класса FallbackModel.
    """
    model = FallbackModel(
        openrouter_api_key="test_key",
        openrouter_model="test-model",
        ollama_base_url="http://localhost:11434",
        ollama_model="qwen2.5:0.5b"
    )
    messages = [{"role": "user", "content": "Hello"}]

    _test_openrouter_success(model, messages)
    _test_openrouter_fails_ollama_success(model, messages)
    _test_both_fail(model, messages)
