from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 0


class LanguageDetectorTool:
    """Инструмент для определения языка переданного текста."""

    name = "language_detector"
    description = (
        "Определяет наиболее вероятный язык текста с помощью библиотеки langdetect. "
        "Возвращает код языка и список вероятностей."
    )

    def use(self, text: str) -> str:
        """
        Определяет язык текста.

        Args:
            text: текст для определения языка.

        Returns:
            Строка с кодом языка и вероятностями.
        """

        if not isinstance(text, str):
            return "Ошибка: входные данные должны быть строкой."

        text = text.strip()

        if not text:
            return "Ошибка: передан пустой текст."

        try:
            probabilities = detect_langs(text)
            language = probabilities[0].lang

            probabilities_str = ", ".join(
                f"{item.lang}: {item.prob:.3f}"
                for item in probabilities
            )

            return (
                f"Определённый язык: {language}\n"
                f"Вероятности: {probabilities_str}"
            )

        except LangDetectException:
            return "Ошибка: язык текста определить не удалось."
