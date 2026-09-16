from unittest.mock import Mock, patch

from llm_agent.tool_weather import WeatherTool


def _response(json_data):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = json_data
    return response


def test_get_coordinates_returns_city_data():
    tool = WeatherTool()

    with patch("llm_agent.tool_weather.requests.get") as mock_get:
        mock_get.return_value = _response(
            {
                "results": [
                    {
                        "latitude": 55.7558,
                        "longitude": 37.6176,
                        "name": "Москва",
                    }
                ]
            }
        )

        latitude, longitude, name = tool._get_coordinates("Москва")

    assert latitude == 55.7558
    assert longitude == 37.6176
    assert name == "Москва"


def test_get_current_weather_returns_current_block():
    tool = WeatherTool()
    expected = {
        "temperature_2m": 12.5,
        "apparent_temperature": 10.8,
        "relative_humidity_2m": 65,
        "weather_code": 2,
        "wind_speed_10m": 7.4,
    }

    with patch("llm_agent.tool_weather.requests.get") as mock_get:
        mock_get.return_value = _response({"current": expected})
        result = tool._get_current_weather(55.7558, 37.6176)

    assert result == expected


def test_use_reuses_cache_for_ten_minutes():
    tool = WeatherTool()
    fresh_result = "Погода в городе Москва: 12 °C"

    with patch.object(tool, "_fetch_weather", return_value=fresh_result) as fetch:
        first = tool.use("Москва")
        second = tool.use("москва")

    assert first == fresh_result
    assert second == fresh_result
    assert fetch.call_count == 1


def test_expired_cache_triggers_new_request():
    tool = WeatherTool()
    tool._cache["москва"] = (
        100.0,
        "старые данные",
    )

    with patch("llm_agent.tool_weather.time.monotonic", return_value=701.0):
        with patch.object(tool, "_fetch_weather", return_value="новые данные") as fetch:
            result = tool.use("Москва")

    assert result == "новые данные"
    assert fetch.call_count == 1


def test_unknown_city_returns_readable_error():
    tool = WeatherTool()

    with patch("llm_agent.tool_weather.requests.get") as mock_get:
        mock_get.return_value = _response({"results": []})
        result = tool.use("НесуществующийГород")

    assert result.startswith("Ошибка получения погоды")
    assert "не найден" in result
