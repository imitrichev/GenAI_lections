from llm_agent.core_v2 import LLMAgent

def main():
    """Основная функция для запуска агента."""
    print("LLM-агент с инструментами (Калькулятор, Поиск, Геокодирование)")
    print("-" * 70)

    # Используем локальную модель, как в задании
    agent = LLMAgent(local=True, ollama_model="qwen2.5:3b")
    
    # ЗАПРОС, КОТОРЫЙ ПРЯМО ТРЕБУЕТ ВАШ ИНСТРУМЕНТ ГЕОКОДИРОВАНИЯ
    query = "Какие координаты у города Санкт-Петербург? А также, что находится по координатам 55.75, 37.62?"

    print(f"Ваш запрос: {query}")
    print("-" * 70)

    response = agent.process_query(query)

    print("\n" + "=" * 70)
    print("Финальный ответ агента:\n")
    print(response)
    print("=" * 70)

if __name__ == "__main__":
    main()