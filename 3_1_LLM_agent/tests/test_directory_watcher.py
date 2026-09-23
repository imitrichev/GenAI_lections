# tests/test_directory_watcher.py

import unittest
import os
import time
import tempfile
import shutil
from pathlib import Path
from llm_agent.tool_directory_watcher import DirectoryWatcherTool


class TestDirectoryWatcherTool(unittest.TestCase):
    """Тесты для DirectoryWatcherTool"""
    
    def setUp(self):
        """Создаём временную папку для тестов"""
        self.test_dir = tempfile.mkdtemp()
        self.tool = DirectoryWatcherTool()
    
    def tearDown(self):
        """Очищаем после тестов"""
        if self.tool.is_watching:
            self.tool.stop_watching()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)
    
    def _wait_for_events(self, count: int, timeout: float = 3.0):
        """Ждём, пока наберётся нужное количество событий"""
        start = time.time()
        while time.time() - start < timeout:
            if len(self.tool.get_events_list()) >= count:
                return True
            time.sleep(0.1)
        return False
    
    def test_start_watching_valid_path(self):
        """Тест 1: Запуск отслеживания валидной папки"""
        result = self.tool.use("start", self.test_dir)
        self.assertIn("Начато отслеживание", result)
        self.assertTrue(self.tool.is_watching)
    
    def test_start_watching_invalid_path(self):
        """Тест 2: Запуск отслеживания несуществующей папки"""
        result = self.tool.use("start", "C:/nonexistent/path/12345")
        self.assertIn("не существует", result)
        self.assertFalse(self.tool.is_watching)
    
    def test_detect_file_creation(self):
        """Тест 3: Обнаружение создания файла"""
        self.tool.use("start", self.test_dir)
        time.sleep(0.5)
        
        test_file = os.path.join(self.test_dir, "test.txt")
        with open(test_file, 'w') as f:
            f.write("hello")
        
        self.assertTrue(self._wait_for_events(1), "Событие создания не обнаружено")
        events = self.tool.get_events_list()
        created_events = [e for e in events if e['type'] == 'created']
        self.assertGreater(len(created_events), 0)
    
    def test_stop_watching(self):
        """Тест 4: Остановка отслеживания"""
        self.tool.use("start", self.test_dir)
        self.assertTrue(self.tool.is_watching)
        
        result = self.tool.use("stop")
        self.assertIn("остановлено", result)
        self.assertFalse(self.tool.is_watching)
    
    def test_status_command(self):
        """Тест 5: Команда status"""
        result = self.tool.use("status")
        self.assertIn("не активно", result)
        
        self.tool.use("start", self.test_dir)
        result = self.tool.use("status")
        self.assertIn("активно", result)
    
    def test_unknown_command(self):
        """Тест 6: Неизвестная команда"""
        result = self.tool.use("unknown_cmd")
        self.assertIn("Неизвестная команда", result)


if __name__ == '__main__':
    unittest.main(verbosity=2)