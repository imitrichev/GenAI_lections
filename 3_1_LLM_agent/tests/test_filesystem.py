import os
import shutil
import sys
import unittest
from pathlib import Path

# Добавляем родительскую папку (3_1_LLM_agent) в PATH,
# чтобы Python нашёл пакет llm_agent и модуль tool_filesystem
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from llm_agent.tool_filesystem import FileSystemTool


def test_all():
    """Обёртка для запуска всех тестов одной функцией (требование лабы)."""
    suite = unittest.TestLoader().loadTestsFromTestCase(TestFileSystemTool)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    assert result.wasSuccessful(), "Не все тесты прошли!"


class TestFileSystemTool(unittest.TestCase):

    def setUp(self):
        self.tool = FileSystemTool(sandbox_dir="test_sandbox")

    def tearDown(self):
        if os.path.exists("test_sandbox"):
            shutil.rmtree("test_sandbox")

    def test_write_and_read(self):
        """Запись и чтение файла."""
        result_write = self.tool.write("hello.txt", "Привет мир")
        self.assertIn("OK", result_write)
        result_read = self.tool.read("hello.txt")
        self.assertEqual(result_read, "Привет мир")

    def test_list_files(self):
        """Список файлов в директории."""
        self.tool.write("a.txt", "aaa")
        self.tool.write("b.txt", "bbb")
        files = self.tool.list(".")
        self.assertIn("a.txt", files)
        self.assertIn("b.txt", files)

    def test_delete_file(self):
        """Удаление файла."""
        self.tool.write("delete_me.txt", "test")
        result = self.tool.delete("delete_me.txt")
        self.assertIn("OK", result)
        self.assertFalse(Path("test_sandbox/delete_me.txt").exists())

    def test_sandbox_escape_blocked(self):
        """Выход за пределы песочницы запрещён."""
        with self.assertRaises(PermissionError):
            self.tool.read("../secret.txt")

    def test_read_nonexistent(self):
        """Чтение несуществующего файла возвращает ERROR."""
        result = self.tool.read("no_such_file.txt")
        self.assertIn("ERROR", result)

    def test_use_interface(self):
        """Универсальный интерфейс use()."""
        self.tool.use("write test.txt Содержимое")
        result = self.tool.use("read test.txt")
        self.assertEqual(result, "Содержимое")


if __name__ == "__main__":
    test_all()