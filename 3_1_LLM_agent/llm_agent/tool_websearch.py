# llm_agent/tool_websearch.py

from ddgs import DDGS

class WebSearchTool:
    """Инструмент для поиска информации в интернете с помощью DuckDuckGo."""
    
    name = "web_search"
    description = "Ищет информацию по запросу в интернете и возвращает краткую сводку из первых результатов."

    def use(self, query: str) -> str:
        """
        Выполняет поиск по запросу и возвращает подробные текстовые сниппеты.
        """ 
        try:
            print(f"> Выполняю поиск новостей в DuckDuckGo по запросу: '{query}'")
            
            summaries = []
            with DDGS() as ddgs:
                # Теперь используем специальный источник 'news' для новостей,
                # он часто дает более длинные тексты, чем стандартный 'text'
                search_results = list(ddgs.text(query, max_results=10)) #backend="news"

            if not search_results:
                return f"По запросу '{query}' ничего не найдено."

            print(f"> Найдено {len(search_results)} новостных статей.")
            
            # Собираем подробные описания (сниппеты) в одну сводку
            for i, result in enumerate(search_results, 1):
                title = result.get('title', 'Без заголовка')
                body = result.get('body', 'Нет описания')

                # Убираем слишком длинные тексты, чтобы не перегружать модель
                if len(body) > 1200:
                    body = body[:1200] + "..."

                summaries.append(f"**{i}. {title}**\n{body}\n")
                print(f"> Обработана статья {i}: {title[:50]}...")

            final_summary = "Результаты поиска:\n\n" + "\n".join(summaries)
            return final_summary

        except Exception as e:
            # Это сообщение будет выведено в лог, если ошибка возникнет на самом верхнем уровне
            print(f"> Ошибка при выполнении поиска: {e}")
            return f"Произошла ошибка при поиске информации по запросу '{query}': {e}"
