# main.py

from llm_agent.core_v2 import LLMAgent

def main():
    """Основная функция для запуска агента."""
    print("Простой LLM-агент с инструментами ('Калькулятор', 'Поиск в DuckDuckGo', 'Определение языка')")
    print("-" * 70)

    #agent = LLMAgent(model = "qwen/qwen3-next-80b-a3b-instruct:free")

    agent = LLMAgent(local = True, ollama_model = "qwen3.5:2b") #ollama_base_url = "10.10.34.24:5678"

    query = (
        "Определи язык следующего текста: "
        "Hello, my name is John. I live in London and "
        "I am learning Python programming at university."
    )

    print(f"Ваш запрос: {query}")
    print("-" * 70)

    response = agent.process_query(query)

    print("\n" + "=" * 70)
    print("Финальный ответ агента:\n")
    print(response)
    print("=" * 70)

if __name__ == "__main__":
    main()
