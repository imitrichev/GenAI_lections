# llm_agent/tool_wikipedia.py
import requests


class WikipediaTool:
    """Инструмент для получения краткой выдержки из Wikipedia по запросу."""

    name = "wikipedia"
    description = (
        "Ищет статью в Wikipedia и возвращает краткую выдержку (несколько первых предложений). "
        "Поддерживает русский (ru) и английский (en) языки. "
        "Input: поисковый запрос (например, 'Альберт Эйнштейн')."
    )

    API_URL = "https://{lang}.wikipedia.org/w/api.php"

    def use(self, query: str, lang: str = "ru", sentences: int = 3) -> str:
        """Получает краткую выдержку из Wikipedia."""
        try:
            if not query or not query.strip():
                return "Ошибка: поисковый запрос не может быть пустым."

            if lang not in ("ru", "en"):
                return f"Ошибка: неподдерживаемый язык '{lang}'. Используйте 'ru' или 'en'."

            print(f"> Ищу в Wikipedia ({lang}): '{query}'")

            params = {
                "action": "query",
                "format": "json",
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "redirects": 1,
                "titles": query,
            }
            headers = {"User-Agent": "GenAI-lections/1.0 (educational project)"}

            response = requests.get(
                self.API_URL.format(lang=lang),
                params=params,
                headers=headers,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            pages = data.get("query", {}).get("pages", {})
            for page in pages.values():
                extract = page.get("extract")
                if extract:
                    parts = extract.split(". ")
                    short = ". ".join(parts[:sentences]).strip()
                    if not short.endswith("."):
                        short += "."
                    print(f"> Найдена статья, возвращаю {len(parts[:sentences])} предл.")
                    return short

            print("> Статья не найдена")
            return f"Статья по запросу '{query}' не найдена в Wikipedia ({lang})."

        except requests.exceptions.Timeout:
            return "Ошибка: превышено время ожидания ответа от Wikipedia."
        except requests.exceptions.RequestException as e:
            return f"Ошибка сети при обращении к Wikipedia: {e}"
        except Exception as e:
            return f"Произошла ошибка при поиске в Wikipedia: {e}"
