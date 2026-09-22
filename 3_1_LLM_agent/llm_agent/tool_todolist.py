import json
import os
import re
from typing import Optional


class TodoListTool:
    """Инструмент для управление списком задач (добавление, удаление, отметка, просмотр) с хранением в JSON."""
    
    name = "todo_list"
    description = "Управляет списком задач. Пример: (add, купить хлеб). Поддерживает команды: add, delete, mark, list"

    def __init__(self, storage_file: str = "todo_list.json"):
        self.storage_file = storage_file
        self.tasks: list[dict] = []
        self.next_id: int = 1
        self._load()

    def use(self, string: str) -> str:
        """
        Принимает строку с математическим выражением и возвращает результат.
        
        Args:
            string (str): Строка для парсинга, например, "(add, купить молоко)".
            
        Returns:
            str: Строка с результатом обработки команды.
        """
        parsed = self._parse(string)
        if parsed is None:
            return "Ошибка: неверный формат команды. Ожидается: (command, task_name / task_id)"

        command, argument = parsed

        if command == "add":
            return self._add(argument)
        elif command == "delete":
            return self._delete(argument)
        elif command == "mark":
            return self._mark(argument)
        elif command == "list":
            return self._list()
        else:
            return f"Ошибка: неизвестная команда '{command}'. Доступные: add, delete, list, mark"


    def _parse(self, string: str) -> Optional[tuple[str, str]]:
        string = string.strip()

        match = re.match(r"^\(\s*(.*?)\s*\)$", string, re.DOTALL)
        if not match:
            return None

        inner = match.group(1)
        parts = inner.split(",", 1)
        command = parts[0].strip().lower()

        if not command:
            return None

        argument = parts[1].strip() if len(parts) > 1 else ""
        return command, argument

    # ---------- Команды ----------

    def _add(self, task_name: str) -> str:
        if not task_name:
            return "Ошибка: не указано имя задачи"

        task_id = self.next_id
        self.next_id += 1
        self.tasks.append({"id": task_id, "name": task_name, "marked": False})
        self._save()
        return f"Задача добавлена: [{task_id}] {task_name}"

    def _delete(self, argument: str) -> str:
        task_id = self._resolve_id(argument)
        if task_id is None:
            return f"Ошибка: задача не найдена ({argument})"

        for i, task in enumerate(self.tasks):
            if task["id"] == task_id:
                removed = self.tasks.pop(i)
                self._save()
                return f"Задача удалена: [{task_id}] {removed['name']}"

        return f"Ошибка: задача с id={task_id} не найдена"

    def _mark(self, argument: str) -> str:
        task_id = self._resolve_id(argument)
        if task_id is None:
            return f"Ошибка: задача не найдена ({argument})"

        for task in self.tasks:
            if task["id"] == task_id:
                task["marked"] = not task["marked"]
                self._save()
                status = "выполнена" if task["marked"] else "не выполнена"
                return f"Задача [{task_id}] '{task['name']}' отмечена как {status}"

        return f"Ошибка: задача с id={task_id} не найдена"

    def _list(self) -> str:
        if not self.tasks:
            return "Список задач пуст."

        lines = ["Список задач:"]
        for task in self.tasks:
            check = "[x]" if task["marked"] else "[ ]"
            lines.append(f"{check} [{task['id']}] {task['name']}")
        return "\n".join(lines)

    def _resolve_id(self, argument: str) -> Optional[int]:
        """Если аргумент - число, ищем по id, иначе - по имени."""
        if not argument:
            return None

        if argument.isdigit():
            task_id = int(argument)
            for task in self.tasks:
                if task["id"] == task_id:
                    return task_id
            return None

        for task in self.tasks:
            if task["name"] == argument:
                return task["id"]

        return None

    def _load(self) -> None:
        if not os.path.exists(self.storage_file):
            return

        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        if not isinstance(data, dict):
            return

        self.tasks = data.get("tasks", [])
        self.next_id = data.get("next_id", 1)

    def _save(self) -> None:
        data = {"tasks": self.tasks, "next_id": self.next_id}
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)