# tests/test_tool_excelreader.py

import os
import tempfile
import pandas as pd
import pytest
from llm_agent.tool_excelreader import ExcelReaderTool


# 1. Юнит-тест: Проверка корректного чтения файла и лимита строк
def test_excel_reader_valid_data():
    """Проверяет чтение существующего Excel-файла с ограничением вывода строк."""
    tool = ExcelReaderTool()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "test.xlsx")
        df = pd.DataFrame({
            "Name": ["Alice", "Bob", "Charlie", "David"],
            "Score": [10, 20, 30, 40]
        })
        df.to_excel(file_path, index=False)

        # Вызываем инструмент с лимитом в 2 строки
        result = tool.use(file_path, rows_limit=2)

        assert "Данные из файла" in result
        assert "Alice" in result
        assert "Bob" in result
        assert "Charlie" not in result  # Третья строка не должна попадать в вывод


# 2. Юнит-тест: Проверка обработки пустого Excel-файла
def test_excel_reader_empty_file():
    """Проверяет корректную реакцию инструмента на пустой файл."""
    tool = ExcelReaderTool()

    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "empty.xlsx")
        df = pd.DataFrame()
        df.to_excel(file_path, index=False)

        result = tool.use(file_path)

        assert "не содержит данных" in result


# 3. Юнит-тест: Проверка обработки несуществующего пути
def test_excel_reader_invalid_path():
    """Проверяет обработку исключения при передаче неверного пути к файлу."""
    tool = ExcelReaderTool()
    result = tool.use("non_existent_excel_file_123.xlsx")

    assert "Произошла ошибка при чтении Excel-файла" in result


# 4. Вызов всех трех тестов внутри отдельной тестовой функции
# (Прямое требование пункта 4 задания)
def test_all_excel_reader_features():
    """Комплексный запуск всех функций проверки класса ExcelReaderTool."""
    test_excel_reader_valid_data()
    test_excel_reader_empty_file()
    test_excel_reader_invalid_path()