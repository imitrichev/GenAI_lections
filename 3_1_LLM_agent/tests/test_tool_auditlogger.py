import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_agent.tool_auditlogger import AuditLogger


class TestAuditLogger(unittest.TestCase):
    """Юнит-тесты класса AuditLogger."""

    def setUp(self):
        self.logger = AuditLogger()

    def test_log_request_creates_entry_with_query(self):
        # log_request должен создать запись с типом "request", сохранить
        # текст запроса, проставить timestamp и добавить запись в лог.
        entry = self.logger.log_request("Сколько будет 2 + 2?")
        self.assertEqual(entry["event_type"], "request")
        self.assertEqual(entry["query"], "Сколько будет 2 + 2?")
        self.assertIn("timestamp", entry)
        self.assertEqual(len(self.logger.get_log()), 1)

    def test_log_plan_stores_full_plan(self):
        # log_plan должен сохранить план целиком (список шагов) без искажений.
        plan = [{"action": "calculator", "input": "2 + 2"}]
        entry = self.logger.log_plan(plan)
        self.assertEqual(entry["event_type"], "plan")
        self.assertEqual(entry["plan"], plan)

    def test_log_tool_result_records_tool_input_and_result(self):
        # log_tool_result должен сохранить имя инструмента, его вход и результат.
        entry = self.logger.log_tool_result("calculator", "2 + 2", "4")
        self.assertEqual(entry["event_type"], "tool_result")
        self.assertEqual(entry["tool"], "calculator")
        self.assertEqual(entry["input"], "2 + 2")
        self.assertEqual(entry["result"], "4")

    def test_log_final_response_records_response(self):
        # log_final_response должен сохранить итоговый ответ агента.
        entry = self.logger.log_final_response("Ответ: 4")
        self.assertEqual(entry["event_type"], "final_response")
        self.assertEqual(entry["response"], "Ответ: 4")

    def test_get_log_preserves_order_of_events(self):
        # get_log должен возвращать записи в том же порядке, в котором
        # они логировались — это важно для последующего аудита.
        self.logger.log_request("запрос")
        self.logger.log_plan([])
        self.logger.log_final_response("ответ")
        event_types = [entry["event_type"] for entry in self.logger.get_log()]
        self.assertEqual(event_types, ["request", "plan", "final_response"])

    def test_to_json_returns_valid_json_matching_entries(self):
        # to_json должен сериализовать накопленный лог в валидный JSON,
        # который при парсинге обратно совпадает с get_log().
        self.logger.log_request("запрос")
        self.logger.log_final_response("ответ")
        parsed = json.loads(self.logger.to_json())
        self.assertEqual(parsed, self.logger.get_log())

    def test_clear_empties_the_log(self):
        # clear должен полностью очищать накопленный в памяти лог.
        self.logger.log_request("запрос")
        self.logger.clear()
        self.assertEqual(self.logger.get_log(), [])

    def test_log_file_persists_entries_as_json_lines(self):
        # При указании log_file каждая запись должна дописываться в файл
        # отдельной строкой в формате JSON Lines (по одной записи на строку).
        log_path = "test_audit_log.jsonl"
        if os.path.exists(log_path):
            os.remove(log_path)
        try:
            file_logger = AuditLogger(log_file=log_path)
            file_logger.log_request("запрос")
            file_logger.log_final_response("ответ")

            with open(log_path, "r", encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]

            self.assertEqual(len(lines), 2)
            self.assertEqual(lines[0]["event_type"], "request")
            self.assertEqual(lines[1]["event_type"], "final_response")
        finally:
            if os.path.exists(log_path):
                os.remove(log_path)


def run_all_tests():
    """Отдельная тестовая функция, запускающая все юнит-тесты AuditLogger."""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAuditLogger)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
