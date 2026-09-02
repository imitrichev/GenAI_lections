# tests/test_tool_qrcode.py

import base64
import os

import pytest

from llm_agent.tool_qrcode import QRCodeTool

# Сигнатура, с которой начинается любой корректный PNG-файл.
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@pytest.fixture
def tool():
    """Экземпляр инструмента с настройками по умолчанию."""
    return QRCodeTool()


# =====================================================================
# ЮНИТ-ТЕСТЫ ОТДЕЛЬНЫХ ФУНКЦИЙ КЛАССА QRCodeTool
# =====================================================================

def test_parse_query_splits_text_and_path(tool):
    """parse_query() должен отделять текст от пути к файлу."""
    text, file_path = tool.parse_query("  https://example.com | out/qr.png  ")
    assert text == "https://example.com"
    assert file_path == "out/qr.png"

    # Без разделителя '|' путь не указан
    text, file_path = tool.parse_query("просто текст")
    assert text == "просто текст"
    assert file_path is None

    # Пустой текст недопустим
    with pytest.raises(ValueError):
        tool.parse_query("   ")


def test_to_base64_returns_valid_png(tool):
    """to_base64() должен возвращать декодируемую base64-строку с PNG."""
    encoded = tool.to_base64("https://example.com")

    assert isinstance(encoded, str)
    raw = base64.b64decode(encoded, validate=True)
    assert raw.startswith(PNG_SIGNATURE)

    # Разный текст даёт разные картинки
    assert encoded != tool.to_base64("другой текст")


def test_save_to_file_creates_image(tool, tmp_path):
    """save_to_file() должен создавать непустой файл и вернуть абсолютный путь."""
    target = tmp_path / "nested" / "qr.png"

    saved_path = tool.save_to_file("Лабораторная работа 1", str(target))

    assert saved_path == os.path.abspath(str(target))
    assert os.path.isfile(saved_path)
    with open(saved_path, "rb") as f:
        assert f.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE


def test_save_to_file_rejects_bad_extension(tool, tmp_path):
    """save_to_file() не должен сохранять файлы с неподдерживаемым расширением."""
    with pytest.raises(ValueError):
        tool.save_to_file("текст", str(tmp_path / "qr.txt"))

    with pytest.raises(ValueError):
        tool.save_to_file("текст", "   ")


def test_make_image_size_grows_with_box_size():
    """make_image() должен учитывать параметры, заданные в конструкторе."""
    small = QRCodeTool(box_size=2).make_image("тест")
    big = QRCodeTool(box_size=8).make_image("тест")

    assert big.size[0] > small.size[0]
    assert big.size[1] > small.size[1]

    # Некорректные параметры конструктора отвергаются сразу
    with pytest.raises(ValueError):
        QRCodeTool(box_size=0)
    with pytest.raises(ValueError):
        QRCodeTool(border=-1)
    with pytest.raises(ValueError):
        QRCodeTool(error_correction="Z")


def test_use_handles_both_modes_and_errors(tool, tmp_path):
    """use() должен работать в обоих режимах и не падать на плохом вводе."""
    target = tmp_path / "from_use.png"

    file_answer = tool.use(f"https://example.com | {target}")
    assert "сохранён в файл" in file_answer
    assert os.path.isfile(str(target))

    base64_answer = tool.use("текст без файла")
    assert "base64" in base64_answer

    error_answer = tool.use("")
    assert error_answer.startswith("Ошибка")


def test_validation_rejects_bad_input(tool):
    """Некорректный тип и слишком длинный текст должны отвергаться."""
    with pytest.raises(ValueError):
        tool.parse_query(42)

    from llm_agent.tool_qrcode import MAX_TEXT_LENGTH

    with pytest.raises(ValueError):
        tool.make_image("a" * (MAX_TEXT_LENGTH + 1))


# =====================================================================
# ОБЩАЯ ТЕСТОВАЯ ФУНКЦИЯ (вызывает все юнит-тесты подряд)
# =====================================================================

def test_all_qrcode_unit_tests(tmp_path):
    """Запускает все юнит-тесты класса QRCodeTool внутри одной функции."""
    qr_tool = QRCodeTool()

    test_parse_query_splits_text_and_path(qr_tool)
    test_to_base64_returns_valid_png(qr_tool)
    test_save_to_file_creates_image(qr_tool, tmp_path / "all_1")
    test_save_to_file_rejects_bad_extension(qr_tool, tmp_path / "all_2")
    test_make_image_size_grows_with_box_size()
    test_use_handles_both_modes_and_errors(qr_tool, tmp_path / "all_3")
    test_validation_rejects_bad_input(qr_tool)
