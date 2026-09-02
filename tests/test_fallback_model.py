import unittest
import sys
import os
from unittest.mock import patch, Mock
import requests

# Добавляем путь к папке с модулем
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '3_1_LLM_agent'))
from fallback_model import FallbackModel


def is_ollama_available(model="hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_S", url="http://localhost:11434", timeout=5):
    """
    Проверяет доступность Ollama и наличие указанной модели.
    """
    try:
        # Проверяем, отвечает ли сервер
        response = requests.get(f"{url}/api/tags", timeout=timeout)
        if response.status_code != 200:
            return False
        # Проверяем, есть ли модель в списке
        models = response.json().get("models", [])
        for m in models:
            if m.get("name") == model:
                return True
        return False
    except:
        return False


class TestFallbackModel(unittest.TestCase):

    def setUp(self):
        self.model = FallbackModel(
            openrouter_api_key="test_key",
            openrouter_model="test/model",
            ollama_model="hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_S",
            ollama_url="http://localhost:11434",
            timeout=5,
            retries=1
        )

    # --- Юнит-тесты с моками (обязательные) ---

    @patch('fallback_model.requests.post')
    def test_openrouter_success(self, mock_post):
        """Тест 1: успешный вызов OpenRouter."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello from OpenRouter"}}]
        }
        mock_post.return_value = mock_response

        result = self.model.generate("Test prompt")
        self.assertEqual(result, "Hello from OpenRouter")
        mock_post.assert_called_once()

    @patch('fallback_model.requests.post')
    def test_fallback_to_ollama(self, mock_post):
        """Тест 2: при ошибке OpenRouter происходит переключение на Ollama."""
        # Первый вызов (OpenRouter) – ошибка
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        # Второй вызов (Ollama) – успех
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Hello from Ollama"}

        mock_post.side_effect = [mock_response_fail, mock_response_success]

        result = self.model.generate("Test prompt")
        self.assertEqual(result, "Hello from Ollama")
        self.assertEqual(mock_post.call_count, 2)

    @patch('fallback_model.requests.post')
    def test_both_unavailable(self, mock_post):
        """Тест 3: оба API недоступны – возвращается None."""
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_post.side_effect = [mock_response_fail, mock_response_fail]

        result = self.model.generate("Test prompt")
        self.assertIsNone(result)
        self.assertEqual(mock_post.call_count, 2)

    @patch('fallback_model.requests.post')
    def test_primary_recovery_after_reset(self, mock_post):
        """Тест 4: сброс флага primary_available."""
        # Сначала OpenRouter недоступен
        mock_response_fail = Mock()
        mock_response_fail.status_code = 500
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {"response": "Ollama response"}
        mock_post.side_effect = [mock_response_fail, mock_response_success]

        result1 = self.model.generate("Test")
        self.assertEqual(result1, "Ollama response")

        # Сбрасываем флаг
        self.model.reset_primary()

        # Теперь OpenRouter снова доступен
        mock_response_or = Mock()
        mock_response_or.status_code = 200
        mock_response_or.json.return_value = {
            "choices": [{"message": {"content": "OpenRouter response"}}]
        }
        mock_post.side_effect = [mock_response_or]

        result2 = self.model.generate("Test")
        self.assertEqual(result2, "OpenRouter response")

    # --- Интеграционный тест с реальной Ollama (дополнительный) ---

    @unittest.skipIf(not is_ollama_available(), "Ollama not available or model not found")
    def test_real_ollama_generate(self):
        """Тест 5: реальный вызов Ollama (интеграционный)."""
        # Создаём экземпляр с реальными параметрами
        real_model = FallbackModel(
            openrouter_api_key="dummy",  # не будет использоваться, т.к. мы сразу идём в Ollama
            openrouter_model="dummy",
            ollama_model="hf.co/unsloth/Qwen3.5-4B-GGUF:Q4_K_S",
            ollama_url="http://localhost:11434",
            timeout=120,
            retries=1
        )
        # Принудительно переключаемся на Ollama, чтобы не пытаться вызвать OpenRouter
        real_model._primary_available = False

        result = real_model.generate("Напиши слово 'Привет' на русском")
        self.assertIsNotNone(result)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_process_query_returns_string(self):
        """Тест 6: метод process_query всегда возвращает строку."""
        # Мокаем generate, чтобы он возвращал None
        with patch.object(self.model, 'generate', return_value=None):
            result = self.model.process_query("Test")
            self.assertEqual(result, "Извините, не удалось получить ответ ни от одного из API.")

        with patch.object(self.model, 'generate', return_value="OK"):
            result = self.model.process_query("Test")
            self.assertEqual(result, "OK")


if __name__ == '__main__':
    unittest.main()