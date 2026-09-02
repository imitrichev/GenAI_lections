from pydantic import BaseModel
from spellchecker import SpellChecker


class SpellResponse(BaseModel):
    raw_data: str
    suggestion: str | None


class SpellCheckerTool:
    """Инструмент для проверки орфографии текста."""

    name = "spell_check"
    description = "Проверяет орфографию текста и предлагает исправления. Поддерживает русский и английский языки."

    @staticmethod
    def get_message_language(message: str) -> str:
        """Получение языка"""

        return 'en' if message.isascii() else 'ru'

    def get_message_suggestions(self, message: str) -> list[SpellResponse]:
        if not message.strip():
            return []

        language = self.get_message_language(message)
        spell_checker = SpellChecker(language=language)

        return [
            SpellResponse(
                raw_data=word,
                suggestion=spell_checker.correction(word)
            ) for word in message.split()
        ]

    def use(self, text: str) -> str:
        """
        Проверяет орфографию текста и возвращает результаты.

        Args:
            text (str): Текст для проверки.

        Returns:
            str: Строка с результатами проверки.
        """
        if not text or not text.strip():
            return "Текст для проверки не предоставлен."

        suggestions = self.get_message_suggestions(text)

        if not suggestions:
            return f"Текст не содержит слов для проверки: '{text}'"

        corrections = []
        for item in suggestions:
            if item.suggestion and item.suggestion != item.raw_data:
                corrections.append(f"'{item.raw_data}' -> '{item.suggestion}'")

        if not corrections:
            return f"Ошибок не найдено. Все слова корректны: '{text}'"

        return f"Найденные ошибки:\n" + "\n".join(corrections)
