import unittest
from unittest.mock import patch
from tool_geocoding import GeocodingTool

class TestGeocodingTool(unittest.TestCase):
    def setUp(self):
        self.tool = GeocodingTool(user_agent="TestAgent/1.0")

    @patch('tool_geocoding.requests.get')
    def test_use_coords_success(self, mock_get):
        mock_get.return_value.json.return_value = [{"lat": "55.7539", "lon": "37.6208"}]
        mock_get.return_value.raise_for_status = lambda: None
        
        # Тестируем именно тот метод, который вызывает агент
        result = self.tool.use("coords: Москва")
        
        self.assertIn("55.7539", result)
        self.assertIn("37.6208", result)

    @patch('tool_geocoding.requests.get')
    def test_use_address_success(self, mock_get):
        mock_get.return_value.json.return_value = {"display_name": "Красная площадь, Москва"}
        mock_get.return_value.raise_for_status = lambda: None
        
        result = self.tool.use("address: 55.7539, 37.6208")
        
        self.assertEqual(result, "Красная площадь, Москва")

    def test_use_invalid_format(self):
        # Тест без моков, проверяем логику парсинга
        result = self.tool.use("просто какой-то текст")
        self.assertIn("Неверный формат", result)

if __name__ == '__main__':
    unittest.main()