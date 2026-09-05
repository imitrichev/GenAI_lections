# main.py

from core_v2 import LLMAgent, ProviderConfig
from settings import UserConfig


def main():
    """Основная функция для запуска агента."""
    print("Простой LLM-агент с инструментами ('Калькулятор', 'Поиск в DuckDuckGo')")
    print("-" * 70)

    config = UserConfig()

    if config.use_local_model:
        provider = ProviderConfig.ollama(config.base_url, config.ollama_model)
    else:
        provider = ProviderConfig.openrouter(config.model)

    agent = LLMAgent(provider=provider)

    # agent = LLMAgent(model = "gpt-5.4-mini")
    # agent = LLMAgent(model = "grok4.1-fast")

    # Примеры запросов
    # query = "Сколько будет (5 + 3) * 2?"
    # query = "Какая погода в Москве?"
    query = input()

    print("-" * 70)

    response = agent.process_query(query)

    print("\n" + "=" * 70)
    print("Финальный ответ агента:\n")
    print(response)
    print("=" * 70)


if __name__ == "__main__":
    main()
