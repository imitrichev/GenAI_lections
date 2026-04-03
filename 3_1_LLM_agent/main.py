# main.py

from llm_agent.core_v2 import LLMAgent

def main():
    """Основная функция для запуска агента."""
    print("Простой LLM-агент с инструментами ('Калькулятор', 'Поиск в DuckDuckGo')")
    print("-" * 70)

    agent = LLMAgent(model = "qwen/qwen3.6-plus:free")

    #agent = LLMAgent(local = True, ollama_base_url = "10.10.34.24:5678", ollama_model = "qwen3:4b")

    #agent = LLMAgent(model = "grok4.1-fast")
    
    # Примеры запросов
    # query = "Сколько будет (5 + 3) * 2?"
    # query = "Какая погода в Москве?"
    query = "Сколько будет (5 + 3) * 2? А также, кто выиграл последний матч Спартак-Динамо?"

    print(f"Ваш запрос: {query}")
    print("-" * 70)

    response = agent.process_query(query)

    print("\n" + "=" * 70)
    print("Финальный ответ агента:\n")
    print(response)
    print("=" * 70)

if __name__ == "__main__":
    main()
