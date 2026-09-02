import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_agent.core_v2 import LLMAgent

OLLAMA_MODEL = os.environ.get("OLLAMA_TEST_MODEL", "qwen3:0.6b")


class TestLLMAgentWithOllama(unittest.TestCase):
    """
    Интеграционные тесты LLMAgent с реальной моделью в локальном Ollama.
    Требуют запущенный 'ollama serve' и стянутую модель OLLAMA_MODEL.
    """

    @classmethod
    def setUpClass(cls):
        cls.agent = LLMAgent(local=True, ollama_model=OLLAMA_MODEL)
        if not cls.agent.test_ollama_connection():
            raise unittest.SkipTest(
                f"Ollama недоступен по адресу {cls.agent.ollama_base_url}. "
                f"Запустите 'ollama serve' и 'ollama pull {OLLAMA_MODEL}' перед запуском теста."
            )

    def test_ollama_connection_is_available(self):
        # Проверяем, что агент действительно видит запущенный Ollama-сервер.
        self.assertTrue(self.agent.test_ollama_connection())

    def test_process_query_returns_non_empty_answer(self):
        # Прогоняем реальный запрос через модель и проверяем, что агент
        # вернул непустой текстовый ответ.
        response = self.agent.process_query("Сколько будет 2 + 2? Ответь одним числом.")
        self.assertIsInstance(response, str)
        self.assertTrue(len(response.strip()) > 0)

    def test_process_query_logs_full_audit_trail(self):
        # Проверяем, что при работе с реальной моделью AuditLogger фиксирует
        # весь цикл: запрос -> план -> финальный ответ, в правильном порядке.
        self.agent.audit_logger.clear()
        self.agent.process_query("Сколько будет 10 умножить на 3?")

        event_types = [entry["event_type"] for entry in self.agent.audit_logger.get_log()]
        self.assertIn("request", event_types)
        self.assertIn("plan", event_types)
        self.assertIn("final_response", event_types)
        self.assertEqual(event_types[0], "request")
        self.assertEqual(event_types[-1], "final_response")


def run_all_tests():
    """Отдельная тестовая функция, запускающая все интеграционные тесты."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestLLMAgentWithOllama)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
