import os
from pathlib import Path
from typing import List


class FileSystemTool:
    """
    Инструмент для безопасной работы с файлами внутри песочницы.
    Поддерживает операции: read, write, list, delete.
    """

    def __init__(self, sandbox_dir: str = "sandbox"):
        self.sandbox = Path(sandbox_dir).resolve()
        self.sandbox.mkdir(parents=True, exist_ok=True)

    def _resolve_safe(self, relative_path: str) -> Path:
        """Проверяет, что путь не выходит за пределы песочницы."""
        if not isinstance(relative_path, str):
            raise TypeError("Путь должен быть строкой")

        target = (self.sandbox / relative_path).resolve()
        if not str(target).startswith(str(self.sandbox)):
            raise PermissionError(
                f"Доступ за пределы песочницы запрещён: {relative_path}"
            )
        return target

    def read(self, relative_path: str) -> str:
        """Читает содержимое файла внутри песочницы."""
        path = self._resolve_safe(relative_path)
        if not path.exists():
            return f"ERROR: Файл не найден: {relative_path}"
        if not path.is_file():
            return f"ERROR: Это не файл: {relative_path}"
        try:
            return path.read_text(encoding="utf-8")
        except Exception as e:
            return f"ERROR: {e}"

    def write(self, relative_path: str, content: str) -> str:
        """Записывает содержимое в файл внутри песочницы."""
        if not isinstance(content, str):
            raise TypeError("Содержимое должно быть строкой")
        path = self._resolve_safe(relative_path)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return f"OK: Записано {len(content)} символов в {relative_path}"
        except Exception as e:
            return f"ERROR: {e}"

    def list(self, relative_path: str = ".") -> List[str]:
        """Возвращает список файлов и папок внутри указанной директории."""
        path = self._resolve_safe(relative_path)
        if not path.exists():
            return [f"ERROR: Директория не найдена: {relative_path}"]
        if not path.is_dir():
            return [f"ERROR: Это не директория: {relative_path}"]
        try:
            return sorted([p.name for p in path.iterdir()])
        except Exception as e:
            return [f"ERROR: {e}"]

    def delete(self, relative_path: str) -> str:
        """Удаляет файл внутри песочницы."""
        path = self._resolve_safe(relative_path)
        if not path.exists():
            return f"ERROR: Файл не найден: {relative_path}"
        if not path.is_file():
            return "ERROR: Удалять можно только файлы"
        try:
            path.unlink()
            return f"OK: Удалён файл {relative_path}"
        except Exception as e:
            return f"ERROR: {e}"

    def use(self, command: str) -> str:
        """
        Универсальный интерфейс для вызова из LLMAgent.
        Формат команды: "<operation> <path> [content]"
        """
        parts = command.split(maxsplit=2)
        if not parts:
            return "ERROR: Пустая команда"

        op = parts[0].lower()
        if op == "read" and len(parts) >= 2:
            return self.read(parts[1])
        if op == "write" and len(parts) >= 3:
            return self.write(parts[1], parts[2])
        if op == "list":
            path = parts[1] if len(parts) >= 2 else "."
            return str(self.list(path))
        if op == "delete" and len(parts) >= 2:
            return self.delete(parts[1])
        return f"ERROR: Неизвестная команда: {command}"