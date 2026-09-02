# llm_agent/tool_auditlogger.py

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


class AuditLogger:
    """
    Логирует действия LLM-агента (запросы пользователя, планы, результаты
    выполнения инструментов, финальные ответы) в структурированном
    JSON-формате для последующего аудита.
    """

    def __init__(self, log_file: Optional[str] = None):
        """
        Args:
            log_file (str, optional): Путь к файлу, в который будет
                дописываться каждая запись лога (в формате JSON Lines).
                Если не задан, записи хранятся только в памяти.
        """
        self.log_file = log_file
        self.entries: List[Dict[str, Any]] = []

    def _record(self, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Формирует и сохраняет одну запись аудита."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **data,
        }
        self.entries.append(entry)
        if self.log_file:
            self._write_to_file(entry)
        return entry

    def _write_to_file(self, entry: Dict[str, Any]) -> None:
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def log_request(self, query: str) -> Dict[str, Any]:
        """Логирует входящий запрос пользователя."""
        return self._record("request", {"query": query})

    def log_plan(self, plan: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Логирует план действий, составленный агентом."""
        return self._record("plan", {"plan": plan})

    def log_tool_result(self, tool_name: str, tool_input: Any, result: str) -> Dict[str, Any]:
        """Логирует результат выполнения одного инструмента."""
        return self._record(
            "tool_result",
            {"tool": tool_name, "input": tool_input, "result": result},
        )

    def log_final_response(self, response: str) -> Dict[str, Any]:
        """Логирует финальный ответ, отправленный пользователю."""
        return self._record("final_response", {"response": response})

    def get_log(self) -> List[Dict[str, Any]]:
        """Возвращает все накопленные записи аудита."""
        return self.entries

    def to_json(self) -> str:
        """Сериализует весь накопленный лог в JSON-строку."""
        return json.dumps(self.entries, ensure_ascii=False, indent=2)

    def clear(self) -> None:
        """Очищает накопленный в памяти лог."""
        self.entries = []
