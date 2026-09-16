import re


class InputFilterTool:
    """
    Инструмент фильтрации вводимых пользователем промптов
    """

    name = "input_filter"
    description = (
        "Проверка текста на наличие конфиденциальных данных "
    )

    # Регулярные выражения для поиска конфиденциальной информации.
    PATTERNS = {
        "api_key": r"(?i)\b(?:sk|api[_-]?key|token|secret)[-_]?[A-Za-z0-9_\-]{16,}\b",
        "password": r"(?i)\b(?:password|passwd|pwd|пароль)\s*[:=]\s*\S+",
        "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        "phone_ru": r"(?:\+7|8)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}",
    }

    MASK = "[REDACTED]"

    def use(self, prompt: str) -> str:
        """
        Проверяет текст и возвращает что найдено и очищенный текст.
        """
        try:
            result = self.analyze(prompt)
        except TypeError as e:
            return f"Ошибка: {e}"

        if result["is_safe"]:
            return "Конфиденциальных данных не обнаружено."

        types_str = ", ".join(sorted({m["type"] for m in result["matches"]}))
        return (
            f"Найдено {len(result['matches'])} сущностей: {types_str}. "
            f"Очищенный текст: {result['clean_text']}"
        )

    def analyze(self, prompt: str) -> dict:
        """
        Структурированный анализ текста.
        """
        if not isinstance(prompt, str):
            raise TypeError(
                "InputFilter ожидает строку, "
                f"получено: {type(prompt).__name__}"
            )

        matches = []
        clean_text = prompt

        for pii_type, pattern in self.PATTERNS.items():
            for m in re.finditer(pattern, prompt):
                matches.append({
                    "type": pii_type,
                    "value": m.group(0),
                    "span": (m.start(), m.end()),
                })
            clean_text = re.sub(pattern, self.MASK, clean_text)

        return {
            "clean_text": clean_text,
            "matches": matches,
            "is_safe": len(matches) == 0,
        }