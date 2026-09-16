import os
import tempfile

import pytest
from PIL import Image, ImageDraw

from llm_agent.core_v2 import LLMAgent
from llm_agent.tool_imagetotext import ImageToTextTool


# =====================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ (требуют живую Ollama — flaky, не в CI)
# =====================================================================
@pytest.mark.integration
def test_calculator_query_live():
    """Реальный запуск агента для проверки математики."""
    agent = LLMAgent(local=True, ollama_model="llama3.2:3b")
    query = "Сколько будет (5 + 3) * 2? Напиши только цифру."

    response = agent.process_query(query)

    assert "16" in response


@pytest.mark.integration
def test_football_query_live():
    """Реальный запуск агента для проверки поиска DuckDuckGo."""
    agent = LLMAgent(local=True, ollama_model="llama3.2:3b")
    query = "Кто выиграл последний матч Спартак-Динамо?"

    response = agent.process_query(query)

    assert "Спартак" in response or "Spartak" in response
    assert "Динамо" in response or "Dynamo" in response


# =====================================================================
# ЮНИТ-ТЕСТЫ ДЛЯ ImageToTextTool (OCR)
# =====================================================================

@pytest.mark.unit
def test_imagetotext_tool_direct():
    """Тест 1: Извлечение текста из сгенерированного изображения."""
    tool = ImageToTextTool()

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        img_path = tmp_file.name

    try:
        # Генерируем большое изображение с крупным текстом
        img = Image.new('RGB', (800, 200), color='white')
        d = ImageDraw.Draw(img)
        d.text((50, 80), "HELLO WORLD", fill='black')
        img.save(img_path)

        result = tool.use(img_path, lang="eng", preprocess=True)

        # Проверяем, что текст извлечен
        assert "HELLO" in result.upper() or "WORLD" in result.upper()
    finally:
        if os.path.exists(img_path):
            os.unlink(img_path)


@pytest.mark.unit
def test_imagetotext_tool_empty_image():
    """Тест 2: Обработка пустого (белого) изображения."""
    tool = ImageToTextTool()

    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        img_path = tmp_file.name

    try:
        img = Image.new('RGB', (300, 100), color='white')
        img.save(img_path)

        result = tool.use(img_path)

        assert "текст не найден" in result or "не найден" in result
    finally:
        if os.path.exists(img_path):
            os.unlink(img_path)


@pytest.mark.unit
def test_imagetotext_tool_invalid_file():
    """Тест 3: Обработка некорректного файла (не изображения)."""
    tool = ImageToTextTool()

    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
        tmp_file.write(b"This is not an image")
        img_path = tmp_file.name

    try:
        result = tool.use(img_path)

        assert "не является корректным изображением" in result or "Ошибка" in result
    finally:
        if os.path.exists(img_path):
            os.unlink(img_path)


@pytest.mark.unit
def test_imagetotext_tool_missing_file():
    """Тест 4: Обработка отсутствующего файла."""
    tool = ImageToTextTool()
    result = tool.use("/path/to/non/existent/image.jpg")

    assert "не найден" in result or "ошибка" in result.lower()


@pytest.mark.unit
def test_imagetotext_tool_empty_source():
    """Тест 5: Обработка пустого source."""
    tool = ImageToTextTool()
    result = tool.use("")

    assert "пустой источник" in result or "Ошибка" in result