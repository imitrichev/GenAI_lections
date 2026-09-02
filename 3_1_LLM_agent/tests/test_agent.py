# tests/test_agent.py
import unittest
import os
import sys

# Добавляем корень проекта в путь (чтобы импорты работали)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_agent.tool_pdfinfo import PDFInfoTool


class TestPDFInfoTool(unittest.TestCase):
    """Юнит-тесты для класса PDFInfoTool"""

    def setUp(self):
        """Создаём экземпляр инструмента перед каждым тестом"""
        self.tool = PDFInfoTool()

    def test_1_name_and_description(self):
        """Тест 1: проверяем name и description"""
        self.assertEqual(self.tool.name, "pdf_info")
        self.assertIsInstance(self.tool.description, str)
        self.assertTrue(len(self.tool.description) > 10)
        self.assertIn("PDF", self.tool.description)

    def test_2_nonexistent_file(self):
        """Тест 2: обработка несуществующего локального файла"""
        result = self.tool.use("/this/path/does/not/exist.pdf")
        self.assertIsInstance(result, str)
        self.assertTrue(
            "ошибка" in result.lower() or "не найден" in result.lower() or "error" in result.lower()
        )

    def test_3_empty_input(self):
        """Тест 3: пустой ввод"""
        result = self.tool.use("")
        self.assertIsInstance(result, str)
        self.assertTrue("ошибка" in result.lower() or "error" in result.lower())

    def test_4_invalid_url(self):
        """Тест 4 (бонус): некорректный URL"""
        result = self.tool.use("https://example.com/not-a-real-pdf")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)


def run_all_tests():
    """
    Отдельная функция, которая запускает все тесты.
    Требование задания: "вызвать их внутри отдельной тестовой функции"
    """
    print("=" * 60)
    print("Запуск всех юнит-тестов PDFInfoTool")
    print("=" * 60)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestPDFInfoTool)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("=" * 60)
    if result.wasSuccessful():
        print("Все тесты успешно пройдены!")
    else:
        print(f"Провалено тестов: {len(result.failures) + len(result.errors)}")
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
