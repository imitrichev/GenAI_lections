# llm_agent/tool_directory_watcher.py

import os
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileSystemEvent


class DirectoryChangeHandler(FileSystemEventHandler):
    """Обработчик событий файловой системы"""
    
    def __init__(self, callback: Optional[Callable] = None):
        self.events: List[Dict] = []
        self.callback = callback
    
    def _record_event(self, event_type: str, event: FileSystemEvent):
        """Записывает событие в список"""
        record = {
            "type": event_type,
            "path": event.src_path,
            "is_directory": event.is_directory,
            "timestamp": datetime.now().isoformat()
        }
        self.events.append(record)
        if self.callback:
            self.callback(record)
    
    def on_created(self, event):
        self._record_event("created", event)
    
    def on_deleted(self, event):
        self._record_event("deleted", event)
    
    def on_modified(self, event):
        self._record_event("modified", event)
    
    def on_moved(self, event):
        self._record_event("moved", event)


class DirectoryWatcherTool:
    """
    Инструмент для отслеживания изменений в папке.
    Использует библиотеку watchdog.
    
    Поддерживает команды:
    - start <path>: начать отслеживание папки
    - stop: остановить отслеживание
    - status: показать текущий статус
    - events: показать последние события
    - clear: очистить список событий
    """
    
    name = "directory_watcher"
    description = "Отслеживает изменения в папке (создание, удаление, изменение файлов)"
    
    def __init__(self):
        self.observer: Optional[Observer] = None
        self.handler: Optional[DirectoryChangeHandler] = None
        self.watched_path: Optional[str] = None
        self.is_watching: bool = False
    
    def use(self, command: str, *args) -> str:
        """Основной метод выполнения команд"""
        # Разбираем строку команды
        parts = command.strip().split(maxsplit=1)
        cmd = parts[0].lower() if parts else ""
        arg = parts[1] if len(parts) > 1 else None
        
        if cmd == "start":
            if not arg and not args:
                return "Ошибка: укажите путь. Пример: start C:\\temp"
            path = arg if arg else args[0]
            return self.start_watching(path)
        
        elif cmd == "stop":
            return self.stop_watching()
        
        elif cmd == "status":
            return self.get_status()
        
        elif cmd == "events":
            limit = int(arg) if arg and arg.isdigit() else 10
            return self.get_events(limit)
        
        elif cmd == "clear":
            return self.clear_events()
        
        else:
            return (f"Неизвестная команда: {cmd}. "
                    f"Доступные: start <path>, stop, status, events [limit], clear")
    
    def start_watching(self, path: str) -> str:
        """Начинает отслеживание папки"""
        if self.is_watching:
            return f"Уже отслеживается: {self.watched_path}. Сначала остановите."
        
        path_obj = Path(path)
        if not path_obj.exists():
            return f"Ошибка: путь '{path}' не существует"
        
        if not path_obj.is_dir():
            return f"Ошибка: '{path}' не является папкой"
        
        try:
            self.handler = DirectoryChangeHandler()
            self.observer = Observer()
            self.observer.schedule(self.handler, str(path_obj), recursive=True)
            self.observer.start()
            self.watched_path = str(path_obj.absolute())
            self.is_watching = True
            
            return f"Начато отслеживание: {self.watched_path}"
        except Exception as e:
            self.is_watching = False
            return f"Ошибка при запуске: {e}"
    
    def stop_watching(self) -> str:
        """Останавливает отслеживание"""
        if not self.is_watching:
            return "Отслеживание не запущено"
        
        try:
            if self.observer:
                self.observer.stop()
                self.observer.join(timeout=5)
            
            path = self.watched_path
            self.observer = None
            self.watched_path = None
            self.is_watching = False
            
            return f"Отслеживание остановлено: {path}"
        except Exception as e:
            return f"Ошибка при остановке: {e}"
    
    def get_status(self) -> str:
        """Возвращает статус"""
        if self.is_watching:
            count = len(self.handler.events) if self.handler else 0
            return (f"Статус: активно\n"
                    f"Папка: {self.watched_path}\n"
                    f"Событий: {count}")
        return "Статус: не активно"
    
    def get_events(self, limit: int = 10) -> str:
        """Возвращает последние события"""
        if not self.handler:
            return "Нет данных. Запустите отслеживание: start <path>"
        
        events = self.handler.events[-limit:]
        if not events:
            return "Событий пока нет"
        
        result = [f"Последние {len(events)} событий:"]
        for e in events:
            result.append(f"  [{e['type']}] {e['path']}")
        
        return "\n".join(result)
    
    def clear_events(self) -> str:
        """Очищает список событий"""
        if not self.handler:
            return "Нет данных для очистки"
        count = len(self.handler.events)
        self.handler.events.clear()
        return f"Очищено событий: {count}"
    
    def get_events_list(self) -> List[Dict]:
        """Возвращает список событий (для тестов)"""
        return self.handler.events if self.handler else []