"""
PDFInfoTool — извлечение информации из PDF-файлов.
Поддерживает:
- получение метаданных (автор, заголовок, дата создания и т.п.)
- подсчёт количества страниц
- извлечение текста (полностью или постранично)
- работу с локальными файлами и URL
"""

import os
import tempfile
import requests
from typing import Dict, List, Optional, Union
from pypdf import PdfReader


class PDFInfoTool:
    """Инструмент для извлечения информации из PDF."""

    def __init__(self, timeout: int = 30, max_pages_text: Optional[int] = None):
        """
        Args:
            timeout: таймаут HTTP-запроса при скачивании PDF по URL (сек).
            max_pages_text: ограничение числа страниц при извлечении текста
                            (None = без ограничения).
        """
        self.timeout = timeout
        self.max_pages_text = max_pages_text

    # Публичный API

    def use(self, source: str) -> str:
        """
        Основной метод (совместим с интерфейсом остальных инструментов агента).
        Возвращает краткую текстовую сводку по PDF.
        """
        try:
            info = self.get_full_info(source)
            parts = [
                f"Источник: {source}",
                f"Страниц: {info['pages']}",
                f"Метаданные: {info['metadata']}",
                "---- Текст (начало) ----",
                info["text"][:2000] if info["text"] else "(текст не извлечён)",
            ]
            return "\n".join(parts)
        except Exception as e:
            return f"Ошибка при обработке PDF: {e}"

    def get_metadata(self, source: str) -> Dict[str, str]:
        """Возвращает словарь метаданных PDF (пустой, если их нет)."""
        reader = self._open_reader(source)
        meta = reader.metadata or {}
        return {str(k): str(v) for k, v in meta.items()}

    def get_page_count(self, source: str) -> int:
        """Возвращает количество страниц."""
        reader = self._open_reader(source)
        return len(reader.pages)

    def extract_text(self, source: str, page: Optional[int] = None) -> str:
        """
        Извлекает текст.
        Если page указан (0-индексация) — только с этой страницы.
        Иначе — со всех страниц (с учётом max_pages_text).
        """
        reader = self._open_reader(source)
        total = len(reader.pages)

        if page is not None:
            if not 0 <= page < total:
                raise IndexError(f"Страница {page} вне диапазона [0, {total - 1}]")
            return reader.pages[page].extract_text() or ""

        limit = total if self.max_pages_text is None else min(total, self.max_pages_text)
        chunks: List[str] = []
        for i in range(limit):
            chunks.append(reader.pages[i].extract_text() or "")
        return "\n".join(chunks)

    def get_full_info(self, source: str) -> Dict[str, Union[int, str, dict]]:
        """Собирает всё сразу: метаданные, число страниц, полный текст."""
        reader = self._open_reader(source)
        total = len(reader.pages)
        limit = total if self.max_pages_text is None else min(total, self.max_pages_text)

        text_chunks = []
        for i in range(limit):
            text_chunks.append(reader.pages[i].extract_text() or "")

        meta = reader.metadata or {}
        return {
            "metadata": {str(k): str(v) for k, v in meta.items()},
            "pages": total,
            "text": "\n".join(text_chunks),
        }

    def _open_reader(self, source: str) -> PdfReader:
        """
        Открывает PDF из локального пути или URL.
        Для URL скачивает во временный файл.
        """
        if self._is_url(source):
            content = self._download(source)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
            tmp.write(content)
            tmp.close()
            return PdfReader(tmp.name)

        if not os.path.isfile(source):
            raise FileNotFoundError(f"Файл не найден: {source}")
        return PdfReader(source)

    @staticmethod
    def _is_url(s: str) -> bool:
        return s.startswith("http://") or s.startswith("https://")

    def _download(self, url: str) -> bytes:
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.content