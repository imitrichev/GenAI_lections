from llm_agent.tool_language_detector import LanguageDetectorTool


def test_detects_english_text():
    tool = LanguageDetectorTool()

    result = tool.use(
        "This is a sufficiently long English sentence for reliable language detection."
    )

    assert "Определённый язык: en" in result
    assert "en:" in result


def test_detects_russian_text():
    tool = LanguageDetectorTool()

    result = tool.use(
        "Это достаточно длинное русское предложение для надежного определения языка."
    )

    assert "Определённый язык: ru" in result
    assert "ru:" in result


def test_rejects_empty_text():
    tool = LanguageDetectorTool()

    result = tool.use("   ")

    assert result == "Ошибка: передан пустой текст."


def test_rejects_non_string_input():
    tool = LanguageDetectorTool()

    result = tool.use(12345)

    assert result == "Ошибка: входные данные должны быть строкой."


def test_handles_text_without_language_features():
    tool = LanguageDetectorTool()

    result = tool.use("12345")

    assert result == "Ошибка: язык текста определить не удалось."


def test_returns_repeatable_result():
    tool = LanguageDetectorTool()
    text = "Bonjour, je voudrais apprendre la programmation en Python."

    first_result = tool.use(text)
    second_result = tool.use(text)

    assert first_result == second_result
    assert "Определённый язык: fr" in first_result
