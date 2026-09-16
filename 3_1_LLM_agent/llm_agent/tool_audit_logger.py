import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional


class AuditLogger:
    """
    Класс для логирования всех действий агента в структурированном JSON формате.
    """
    
    def __init__(self, log_file: str = "audit_logs.json", session_id: Optional[str] = None):
        self.log_file = log_file
        self.logs: List[Dict[str, Any]] = []
        self.session_id = session_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._load_existing_logs()
    
    def _load_existing_logs(self) -> None:
        """Загружает существующие логи из файла."""
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.logs = data.get('logs', [])
            except (json.JSONDecodeError, IOError):
                self.logs = []
    
    def log_request(self, user_query: str, metadata: Optional[Dict] = None) -> Dict:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": "USER_REQUEST",
            "data": {
                "query": user_query,
                "metadata": metadata or {}
            }
        }
        self.logs.append(log_entry)
        self._save_logs()
        return log_entry
    
    def log_plan(self, plan: List[Dict], metadata: Optional[Dict] = None) -> Dict:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": "ACTION_PLAN",
            "data": {
                "plan": plan,
                "steps_count": len(plan),
                "metadata": metadata or {}
            }
        }
        self.logs.append(log_entry)
        self._save_logs()
        return log_entry
    
    def log_tool_execution(self, tool_name: str, tool_input: str, 
                          result: Any, metadata: Optional[Dict] = None) -> Dict:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": "TOOL_EXECUTION",
            "data": {
                "tool_name": tool_name,
                "input": tool_input,
                "result": str(result)[:1000],
                "metadata": metadata or {}
            }
        }
        self.logs.append(log_entry)
        self._save_logs()
        return log_entry
    
    def log_final_response(self, response: str, metadata: Optional[Dict] = None) -> Dict:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": "FINAL_RESPONSE",
            "data": {
                "response": response,
                "response_length": len(response),
                "metadata": metadata or {}
            }
        }
        self.logs.append(log_entry)
        self._save_logs()
        return log_entry
    
    def log_error(self, error_message: str, error_type: str = "GENERAL", 
                  metadata: Optional[Dict] = None) -> Dict:
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "event_type": "ERROR",
            "data": {
                "error_type": error_type,
                "error_message": error_message,
                "metadata": metadata or {}
            }
        }
        self.logs.append(log_entry)
        self._save_logs()
        return log_entry
    
    def _save_logs(self) -> None:
        """Сохраняет логи в файл."""
        print(f" _save_logs вызван! Записей: {len(self.logs)}, файл: {self.log_file}")
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "session_id": self.session_id,
                    "total_logs": len(self.logs),
                    "logs": self.logs
                }, f, ensure_ascii=False, indent=2)
            print(f" Успешно сохранено {len(self.logs)} записей в {self.log_file}")
        except IOError as e:
            print(f"❌ Ошибка сохранения логов: {e}")
        except Exception as e:
            print(f"❌ Неожиданная ошибка: {e}")
    
    def get_logs(self) -> List[Dict]:
        return self.logs
    
    def get_logs_by_event_type(self, event_type: str) -> List[Dict]:
        return [log for log in self.logs if log.get('event_type') == event_type]
    
    def clear_logs(self) -> None:
        self.logs = []
        self._save_logs()
    
    def get_statistics(self) -> Dict:
        stats = {
            "total_logs": len(self.logs),
            "event_types": {}
        }
        for log in self.logs:
            event_type = log.get('event_type', 'UNKNOWN')
            stats["event_types"][event_type] = stats["event_types"].get(event_type, 0) + 1
        return stats