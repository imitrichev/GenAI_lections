import pytest
from llm_agent.tool_input_filter import InputFilterTool


def test_input_filter_all_cases():
    """
    Отдельная тестовая функция, внутри которой вызываются
    все юнит-тесты класса InputFilterTool.
    """
    tool = InputFilterTool()

    # Чистый текст без PII
    result = tool.analyze("Привет! Как дела?")
    assert result["is_safe"] is True
    assert result["matches"] == []
    assert result["clean_text"] == "Привет! Как дела?"

    # Email
    result = tool.analyze("Напишите: ivan.ivanov@gmail.com")
    assert result["is_safe"] is False
    assert any(m["type"] == "email" for m in result["matches"])
    assert "ivan.ivanov@gmail.com" not in result["clean_text"]
    assert "[REDACTED]" in result["clean_text"]

    # API-ключ
    result = tool.analyze("Ключ: sk-proj-abc123XYZ456def789ghi012")
    assert result["is_safe"] is False
    assert any(m["type"] == "api_key" for m in result["matches"])
    assert "sk-proj" not in result["clean_text"]

    # Пароль
    result = tool.analyze("password=Password123")
    assert result["is_safe"] is False
    assert any(m["type"] == "password" for m in result["matches"])
    assert "Password123" not in result["clean_text"]

    # Смешанный ввод
    mixed = "Email: a@b.com, ключ: sk-1234567890abcdefgh, тел: +7 (999) 123-45-67"
    result = tool.analyze(mixed)
    types = {m["type"] for m in result["matches"]}
    assert "email" in types
    assert "api_key" in types
    assert "phone_ru" in types

    with pytest.raises(TypeError):
        tool.analyze(12345)

    # Возврат строки с отчётом
    report = tool.use("Пиши на: user@gmail.com")
    assert isinstance(report, str)
    assert "email" in report
    assert "Очищенный текст" in report

    report = tool.use("Обычный текст")
    assert "Конфиденциальных данных не обнаружено" in report