
import requests


class WikipediaTool:
    """Инструмент для поиска и извлечения кратких выдержек из Wikipedia."""

    name = "wikipedia"
    description = (
        "Ищет информацию в Wikipedia на русском или английском языке "
        "и возвращает краткую выдержку из найденной статьи."
    )

    # Wikipedia рекомендует указывать идентифицирующий User-Agent.
    HEADERS = {
        "User-Agent": (
            "GenAI-Lections/1.0 "
            "(educational project; contact: github.com/prioraLover/GenAI_lections)"
        )
    }

    TIMEOUT = 20

    def use(self, query: str, language: str = "ru") -> str:
        """
        Ищет статью в Wikipedia и возвращает её краткую выдержку.

        Поддерживает два формата:

        1. wiki.use("Искусственный интеллект", "ru")
        2. wiki.use("Искусственный интеллект | ru")
        """

        try:
            # --------------------------------------------------
            # Обработка аргументов
            # --------------------------------------------------

            if "|" in query:
                query, language_from_query = query.rsplit("|", 1)
                query = query.strip()
                language = language_from_query.strip().lower()

            language = language.strip().lower()
            query = query.strip()

            # Проверяем язык.
            if language not in ("ru", "en"):
                return (
                    "Ошибка: поддерживаются только языки "
                    "'ru' (русский) и 'en' (английский)."
                )

            if not query:
                return "Ошибка: поисковый запрос не может быть пустым."

            print(
                f"> Выполняю поиск в Wikipedia ({language}) "
                f"по запросу: '{query}'"
            )

            api_url = f"https://{language}.wikipedia.org/w/api.php"

            # --------------------------------------------------
            # Шаг 1. Поиск статьи
            # --------------------------------------------------

            search_params = {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": 1,
                "format": "json",
                "utf8": 1,
            }

            response = requests.get(
                api_url,
                params=search_params,
                headers=self.HEADERS,
                timeout=self.TIMEOUT,
            )

            response.raise_for_status()

            search_data = response.json()

            search_results = (
                search_data
                .get("query", {})
                .get("search", [])
            )

            if not search_results:
                return (
                    f"В Wikipedia ({language}) ничего не найдено "
                    f"по запросу '{query}'."
                )

            page_title = search_results[0]["title"]

            print(f"> Найдена статья: {page_title}")

            # --------------------------------------------------
            # Шаг 2. Получение краткой выдержки
            # --------------------------------------------------

            extract_params = {
                "action": "query",
                "prop": "extracts",
                "exintro": True,
                "explaintext": True,
                "exchars": 1200,
                "titles": page_title,
                "format": "json",
                "utf8": 1,
            }

            response = requests.get(
                api_url,
                params=extract_params,
                headers=self.HEADERS,
                timeout=self.TIMEOUT,
            )

            response.raise_for_status()

            extract_data = response.json()

            pages = (
                extract_data
                .get("query", {})
                .get("pages", {})
            )

            page = next(iter(pages.values()), {})

            extract = page.get("extract", "").strip()

            if not extract:
                return (
                    f"Статья '{page_title}' найдена, "
                    "но краткая выдержка отсутствует."
                )

            # Ограничиваем длину результата.
            if len(extract) > 1200:
                extract = extract[:1200].rstrip() + "..."

            result = (
                f"Статья Wikipedia: {page_title}\n"
                f"Язык: {language}\n\n"
                f"{extract}"
            )

            print("> Выдержка успешно получена.")

            return result

        except requests.exceptions.Timeout:
            print("> Ошибка: превышено время ожидания запроса.")
            return "Ошибка: Wikipedia не ответила вовремя."

        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code if e.response else None

            print(
                f"> HTTP ошибка Wikipedia: "
                f"{status_code if status_code else e}"
            )

            if status_code == 403:
                return (
                    "Ошибка: Wikipedia отклонила запрос (HTTP 403). "
                    "Попробуйте повторить запрос позже."
                )

            if status_code == 429:
                return (
                    "Ошибка: Wikipedia временно ограничила количество "
                    "запросов (HTTP 429). Попробуйте позже."
                )

            return (
                f"Ошибка при обращении к Wikipedia "
                f"(HTTP {status_code if status_code else 'unknown'})."
            )

        except requests.exceptions.RequestException as e:
            print(f"> Ошибка сети при обращении к Wikipedia: {e}")
            return "Ошибка сети при обращении к Wikipedia."

        except (ValueError, KeyError, TypeError) as e:
            print(f"> Ошибка обработки ответа Wikipedia: {e}")
            return "Ошибка: Wikipedia вернула некорректный ответ."

        except Exception as e:
            print(f"> Неожиданная ошибка при поиске: {e}")
            return (
                f"Произошла ошибка при поиске в Wikipedia "
                f"по запросу '{query}'."
            )

