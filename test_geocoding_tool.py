import unittest
from unittest.mock import patch
from geocoding_tool import GeocodingTool

class TestGeocodingTool(unittest.TestCase):
    def setUp(self):
        self.tool = GeocodingTool(user_agent="TestAgent/1.0")

    @patch('geocoding_tool.requests.get')
    def test_get_coordinates_success(self, mock_get):
        # Настраиваем мок-ответ от API
        mock_get.return_value.json.return_value = [{"lat": "55.7539", "lon": "37.6208"}]
        mock_get.return_value.raise_for_status = lambda: None
        
        result = self.tool.get_coordinates("Москва")
        
        self.assertIsNotNone(result)
        self.assertEqual(result["lat"], 55.7539)
        self.assertEqual(result["lon"], 37.6208)

    @patch('geocoding_tool.requests.get')
    def test_get_address_success(self, mock_get):
        mock_get.return_value.json.return_value = {"display_name": "Красная площадь, Москва"}
        mock_get.return_value.raise_for_status = lambda: None
        
        result = self.tool.get_address(55.7539, 37.6208)
        
        self.assertEqual(result, "Красная площадь, Москва")

    @patch('geocoding_tool.requests.get')
    def test_get_coordinates_not_found(self, mock_get):
        mock_get.return_value.json.return_value = []
        mock_get.return_value.raise_for_status = lambda: None
        
        result = self.tool.get_coordinates("Несуществующее место 12345")
        
        self.assertIsNone(result)

if __name__ == '__main__':
    unittest.main()