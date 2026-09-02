import os
import json
import unittest
import sys

# Добавляем путь к корню проекта
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm_agent.tool_audit_logger import AuditLogger


class TestAuditLogger(unittest.TestCase):
    """Тесты для класса AuditLogger (Вариант 13)"""
    
    def setUp(self):
        """Настройка перед каждым тестом"""
        self.test_log_file = "test_audit_logs.json"
        self.logger = AuditLogger(log_file=self.test_log_file, session_id="test_session")
    
    def tearDown(self):
        """Очистка после каждого теста"""
        if os.path.exists(self.test_log_file):
            os.remove(self.test_log_file)
    
    def test_log_request(self):
        """Тест 1: логирование запроса пользователя"""
        query = "Какая погода в Москве?"
        metadata = {"user_id": "123"}
        
        log_entry = self.logger.log_request(query, metadata)
        
        self.assertEqual(log_entry["event_type"], "USER_REQUEST")
        self.assertEqual(log_entry["data"]["query"], query)
        self.assertEqual(log_entry["session_id"], "test_session")
        self.assertEqual(len(self.logger.get_logs()), 1)
    
    def test_log_plan(self):
        """Тест 2: логирование плана действий"""
        plan = [{"action": "web_search", "input": "погода в Москве"}]
        
        log_entry = self.logger.log_plan(plan)
        
        self.assertEqual(log_entry["event_type"], "ACTION_PLAN")
        self.assertEqual(log_entry["data"]["steps_count"], 1)
        self.assertEqual(len(self.logger.get_logs()), 1)
    
    def test_log_tool_execution(self):
        """Тест 3: логирование выполнения инструмента"""
        log_entry = self.logger.log_tool_execution("calculator", "2+2", "4")
        
        self.assertEqual(log_entry["event_type"], "TOOL_EXECUTION")
        self.assertEqual(log_entry["data"]["tool_name"], "calculator")
        self.assertEqual(log_entry["data"]["result"], "4")
    
    def test_log_final_response(self):
        """Тест 4: логирование финального ответа"""
        response = "Погода в Москве: +20°C"
        
        log_entry = self.logger.log_final_response(response)
        
        self.assertEqual(log_entry["event_type"], "FINAL_RESPONSE")
        self.assertEqual(log_entry["data"]["response"], response)
    
    def test_log_error(self):
        """Тест 5: логирование ошибок"""
        log_entry = self.logger.log_error("Connection timeout", "API_ERROR")
        
        self.assertEqual(log_entry["event_type"], "ERROR")
        self.assertEqual(log_entry["data"]["error_type"], "API_ERROR")
    
    def test_get_statistics(self):
        """Тест 6: получение статистики"""
        self.logger.log_request("query 1")
        self.logger.log_request("query 2")
        self.logger.log_plan([])
        
        stats = self.logger.get_statistics()
        
        self.assertEqual(stats["total_logs"], 3)
        self.assertEqual(stats["event_types"]["USER_REQUEST"], 2)
    
    def test_save_and_load_logs(self):
        """Тест 7: сохранение и загрузка логов из файла"""
        self.logger.log_request("test query")
        self.logger.log_plan([])
        
        new_logger = AuditLogger(log_file=self.test_log_file, session_id="test_session")
        
        self.assertEqual(len(new_logger.get_logs()), 2)
    
    def test_clear_logs(self):
        """Тест 8: очистка логов"""
        self.logger.log_request("test")
        self.assertEqual(len(self.logger.get_logs()), 1)
        
        self.logger.clear_logs()
        self.assertEqual(len(self.logger.get_logs()), 0)
    
    def test_get_logs_by_event_type(self):
        """Тест 9: фильтрация логов по типу события"""
        self.logger.log_request("query 1")
        self.logger.log_plan([])
        self.logger.log_request("query 2")
        self.logger.log_error("err", "ERR")
        
        request_logs = self.logger.get_logs_by_event_type("USER_REQUEST")
        error_logs = self.logger.get_logs_by_event_type("ERROR")
        
        self.assertEqual(len(request_logs), 2)
        self.assertEqual(len(error_logs), 1)


if __name__ == '__main__':
    unittest.main(verbosity=2)