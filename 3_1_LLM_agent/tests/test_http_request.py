from unittest.mock import Mock, patch

from llm_agent.tool_http_request import HTTPRequestTool


def test_parse_get_request():
    """Проверяет разбор GET-запроса с JSON-параметрами."""

    tool = HTTPRequestTool()

    method, url, params = tool._parse_request(
        'GET https://example.com {"name": "Alex"}'
    )

    assert method == "GET"
    assert url == "https://example.com"
    assert params == {"name": "Alex"}


def test_invalid_http_method():
    """Проверяет отклонение неподдерживаемого HTTP-метода."""

    tool = HTTPRequestTool()

    result = tool.use("DELETE https://example.com")

    assert "не поддерживается" in result


def test_invalid_json():
    """Проверяет обработку некорректного JSON."""

    tool = HTTPRequestTool()

    result = tool.use(
        'POST https://example.com {"name": broken}'
    )

    assert "Параметры должны быть валидным JSON" in result


@patch.object(HTTPRequestTool, "_validate_url")
@patch("llm_agent.tool_http_request.requests.get")
def test_get_request(mock_get, mock_validate):
    """Проверяет выполнение GET-запроса."""

    response = Mock()
    response.text = "test response"
    response.raise_for_status.return_value = None

    mock_get.return_value = response

    tool = HTTPRequestTool()

    result = tool.use("GET https://example.com")

    assert result == "test response"

    mock_get.assert_called_once_with(
        "https://example.com",
        params={},
        timeout=10,
        allow_redirects=False,
    )


@patch.object(HTTPRequestTool, "_validate_url")
@patch("llm_agent.tool_http_request.requests.post")
def test_post_request(mock_post, mock_validate):
    """Проверяет выполнение POST-запроса."""

    response = Mock()
    response.text = "created"
    response.raise_for_status.return_value = None

    mock_post.return_value = response

    tool = HTTPRequestTool()

    result = tool.use(
        'POST https://example.com {"name": "Alex"}'
    )

    assert result == "created"

    mock_post.assert_called_once_with(
        "https://example.com",
        json={"name": "Alex"},
        timeout=10,
        allow_redirects=False,
    )
