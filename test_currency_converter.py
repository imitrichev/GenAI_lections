import unittest
import responses
import time
from currency_converter import CurrencyConverterTool


class TestCurrencyConverterTool(unittest.TestCase):
    
    def setUp(self):
        self.converter = CurrencyConverterTool(api_key="test_key")
    
    @responses.activate
    def test_convert_success(self):
        """Тест успешной конвертации."""
        responses.add(
            responses.GET,
            "https://api.exchangerate-api.com/v4/latest/USD",
            json={"base": "USD", "rates": {"EUR": 0.85, "GBP": 0.75}},
            status=200
        )
        result = self.converter.convert(100, "USD", "EUR")
        self.assertAlmostEqual(result, 85.0)
    
    @responses.activate
    def test_cache_used(self):
        """Повторный вызов использует кеш."""
        responses.add(
            responses.GET,
            "https://api.exchangerate-api.com/v4/latest/USD",
            json={"base": "USD", "rates": {"EUR": 0.85}},
            status=200
        )
        self.converter.convert(100, "USD", "EUR")
        self.converter.convert(200, "USD", "EUR")
        self.assertEqual(len(responses.calls), 1)
    
    @responses.activate
    def test_unknown_currency(self):
        """Неизвестная валюта вызывает исключение."""
        responses.add(
            responses.GET,
            "https://api.exchangerate-api.com/v4/latest/USD",
            json={"base": "USD", "rates": {"EUR": 0.85}},
            status=200
        )
        with self.assertRaises(ValueError):
            self.converter.convert(100, "USD", "XYZ")
    
    def test_clear_cache(self):
        """Очистка кеша."""
        self.converter._cache["USD"] = {"rates": {"EUR": 0.85}, "timestamp": time.time()}
        self.converter.clear_cache()
        self.assertEqual(len(self.converter._cache), 0)


def run_all_tests():
    """Отдельная функция для запуска всех тестов."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestCurrencyConverterTool)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    run_all_tests()