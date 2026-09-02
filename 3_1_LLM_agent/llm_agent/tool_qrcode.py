# llm_agent/tool_qrcode.py

import base64
import io
import os

import qrcode
from qrcode.constants import (
    ERROR_CORRECT_L,
    ERROR_CORRECT_M,
    ERROR_CORRECT_Q,
    ERROR_CORRECT_H,
)

# Уровни коррекции ошибок QR-кода: чем выше уровень, тем больше искажений
# кода можно восстановить, но тем плотнее становится картинка.
ERROR_CORRECTION_LEVELS = {
    "L": ERROR_CORRECT_L,   # восстанавливается до 7% данных
    "M": ERROR_CORRECT_M,   # до 15% (значение по умолчанию)
    "Q": ERROR_CORRECT_Q,   # до 25%
    "H": ERROR_CORRECT_H,   # до 30%
}

# Форматы, в которых разрешено сохранять готовый QR-код на диск.
ALLOWED_EXTENSIONS = (".png", ".gif", ".bmp")

# Ограничение на длину входного текста (спецификация QR: 4296 символов
# в алфавитно-цифровом режиме), чтобы не пытаться закодировать книгу.
MAX_TEXT_LENGTH = 4296


class QRCodeTool:
    """Инструмент для генерации QR-кода из текста.

    Умеет два режима работы:
      * сохранить QR-код в файл (если в запросе указан путь);
      * вернуть картинку в виде base64-строки (если путь не указан).
    """

    name = "qrcode"
    description = (
        "Генерирует QR-код из текста. Формат запроса: 'текст | путь/к/файлу.png'. "
        "Если путь не указан, возвращает изображение в виде base64-строки. "
        "Пример: 'https://example.com | qr.png'"
    )

    def __init__(self, box_size: int = 10, border: int = 4, error_correction: str = "M"):
        """Инициализирует инструмент.

        Args:
            box_size (int): Размер одного модуля (чёрного квадратика) в пикселях.
            border (int): Ширина белой рамки в модулях (по стандарту минимум 4).
            error_correction (str): Уровень коррекции ошибок: 'L', 'M', 'Q' или 'H'.

        Raises:
            ValueError: Если переданы некорректные параметры.
        """
        if box_size < 1:
            raise ValueError("box_size должен быть положительным числом")
        if border < 0:
            raise ValueError("border не может быть отрицательным")

        level = str(error_correction).upper()
        if level not in ERROR_CORRECTION_LEVELS:
            raise ValueError(
                f"Неизвестный уровень коррекции ошибок '{error_correction}'. "
                f"Допустимые значения: {', '.join(ERROR_CORRECTION_LEVELS)}"
            )

        self.box_size = box_size
        self.border = border
        self.error_correction = level

    def use(self, query: str) -> str:
        """Основная точка входа инструмента.

        Args:
            query (str): Текст для кодирования и, опционально, путь к файлу
                после разделителя '|', например "https://example.com | qr.png".

        Returns:
            str: Строка с результатом или с описанием ошибки.
        """
        try:
            text, file_path = self.parse_query(query)
            if file_path:
                saved_path = self.save_to_file(text, file_path)
                return f"QR-код для '{text}' сохранён в файл: {saved_path}"

            encoded = self.to_base64(text)
            return f"QR-код для '{text}' (base64, PNG): {encoded}"
        except (ValueError, OSError) as e:
            return f"Ошибка: не могу сгенерировать QR-код по запросу '{query}'. Детали: {e}"

    def parse_query(self, query: str) -> tuple:
        """Разбирает запрос на текст и необязательный путь к файлу.

        Args:
            query (str): Исходный запрос вида "текст" или "текст | файл.png".

        Returns:
            tuple: Пара (text, file_path), где file_path равен None,
                если путь не был указан.

        Raises:
            ValueError: Если текст пустой или слишком длинный.
        """
        if not isinstance(query, str):
            raise ValueError("Запрос должен быть строкой")

        text, separator, file_path = query.partition("|")
        text = text.strip()
        file_path = file_path.strip() if separator else ""

        self._validate_text(text)
        return text, (file_path or None)

    def make_image(self, text: str):
        """Создаёт изображение QR-кода для переданного текста.

        Args:
            text (str): Текст, который нужно закодировать.

        Returns:
            PIL.Image.Image: Готовое изображение QR-кода.

        Raises:
            ValueError: Если текст пустой или превышает лимит длины.
        """
        self._validate_text(text)

        qr = qrcode.QRCode(
            version=None,  # версия подбирается автоматически под объём данных
            error_correction=ERROR_CORRECTION_LEVELS[self.error_correction],
            box_size=self.box_size,
            border=self.border,
        )
        qr.add_data(text)
        qr.make(fit=True)
        return qr.make_image(fill_color="black", back_color="white")

    def to_base64(self, text: str) -> str:
        """Возвращает QR-код в виде base64-строки (PNG).

        Args:
            text (str): Текст, который нужно закодировать.

        Returns:
            str: Изображение PNG, закодированное в base64.
        """
        image = self.make_image(text)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def save_to_file(self, text: str, file_path: str) -> str:
        """Сохраняет QR-код в файл.

        Args:
            text (str): Текст, который нужно закодировать.
            file_path (str): Путь к файлу с расширением .png, .gif или .bmp.

        Returns:
            str: Абсолютный путь к сохранённому файлу.

        Raises:
            ValueError: Если путь пустой или расширение не поддерживается.
            OSError: Если файл не удалось записать.
        """
        if not file_path or not file_path.strip():
            raise ValueError("Путь к файлу не указан")

        file_path = os.path.expanduser(file_path.strip())
        extension = os.path.splitext(file_path)[1].lower()
        if extension not in ALLOWED_EXTENSIONS:
            raise ValueError(
                f"Неподдерживаемое расширение '{extension or 'без расширения'}'. "
                f"Допустимые: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        directory = os.path.dirname(os.path.abspath(file_path))
        os.makedirs(directory, exist_ok=True)

        image = self.make_image(text)
        image.save(file_path)
        return os.path.abspath(file_path)

    def _validate_text(self, text: str) -> None:
        """Проверяет, что текст пригоден для кодирования в QR-код."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Текст для генерации QR-кода не может быть пустым")
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(
                f"Текст слишком длинный: {len(text)} символов, "
                f"максимум — {MAX_TEXT_LENGTH}"
            )
