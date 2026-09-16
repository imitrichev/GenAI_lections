# tests/test_wikipedia.py
from unittest.mock import patch, MagicMock

from llm_agent.tool_wikipedia import WikipediaTool


def _fake_response(payload):
    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = payload
    return resp


def test_use_returns_extract():
    """Успешный сценарий: возвращается короткая выдержка."""
    tool = WikipediaTool()
    payload = {
        "query": {
            "pages": {
                "123": {
                    "extract": (
                        "Альберт Эйнштейн — физик-теоретик. "
                        "Он родился в 1879 году. "
                        "Известен теорией относительности."
                    )
                }
            }
        }
    }
    with patch("llm_agent.tool_wikipedia.requests.get", return_value=_fake_response(payload)):
        result = tool.use("Альберт Эйнштейн", sentences=2)
    assert "Эйнштейн" in result
    assert result.endswith(".")


def test_use_returns_not_found_for_missing_page():
    """Если страницы нет — возвращается сообщение об ошибке."""
    tool = WikipediaTool()
    payload = {"query": {"pages": {"-1": {"missing": ""}}}}
    with patch("llm_agent.tool_wikipedia.requests.get", return_value=_fake_response(payload)):
        result = tool.use("НесуществующаяСтатья12345")
    assert "не найдена" in result.lower()


def test_use_empty_query_does_not_call_http():
    """Пустой запрос — сообщение об ошибке, без HTTP-запроса."""
    tool = WikipediaTool()
    with patch("llm_agent.tool_wikipedia.requests.get") as mock_get:
        result = tool.use("   ")
        mock_get.assert_not_called()
    assert "не может быть пустым" in result.lower()


def test_use_unsupported_language():
    """Неподдерживаемый язык — сообщение об ошибке, без HTTP-запроса."""
    tool = WikipediaTool()
    with patch("llm_agent.tool_wikipedia.requests.get") as mock_get:
        result = tool.use("Python", lang="de")
        mock_get.assert_not_called()
    assert "неподдерживаемый язык" in result.lower()
