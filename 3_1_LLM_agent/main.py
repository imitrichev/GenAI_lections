# main.py
from llm_agent.core_v2 import LLMAgent, ProviderConfig
from llm_agent.settings import UserConfig


def setup_provider() -> ProviderConfig:
    """Подготавливает провайдера перед запросом"""

    config = UserConfig()

    if config.use_local_model:
        provider = ProviderConfig.ollama(base_url=config.base_url, model=config.ollama_model)
    else:
        provider = ProviderConfig.openrouter(config.model)
    return provider


def main():
    """Основная функция для запуска агента"""

    agent = LLMAgent(provider=setup_provider())

    print(f"Простой LLM-агент с инструментами ({agent.show_tools()})")
    print("-" * 70)

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
