# =====================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ (Запускают реальную Ollama / API)
# =====================================================================
# Маркируем как 'integration', чтобы их можно было отключать
# при быстрой проверке.
# =====================================================================

import pytest

from llm_agent.core_v2 import LLMAgent


@pytest.mark.integration
def test_wikipedia_russian_live():
    """
    Интеграционный тест:
    реальная Ollama → LLMAgent → WikipediaTool → Wikipedia API.
    """

    agent = LLMAgent(
        local=True,
        ollama_model="qwen3:0.6b"
    )

    query = "Используй Wikipedia и найди информацию об Альберте Эйнштейне."
    response = agent.process_query(query)

    assert isinstance(response, str)
    assert len(response) > 0

    assert (
        "Эйнштейн" in response
        or "Альберт" in response
        or "Einstein" in response
    )


@pytest.mark.integration
def test_wikipedia_english_live():
    """
    Интеграционный тест:
    реальная Ollama → LLMAgent → WikipediaTool → Wikipedia API.
    """

    agent = LLMAgent(
        local=True,
        ollama_model="qwen3:0.6b"
    )

    query = "Use Wikipedia to find information about Albert Einstein."
    response = agent.process_query(query)

    assert isinstance(response, str)
    assert len(response) > 0

    assert (
        "Einstein" in response
        or "Эйнштейн" in response
        or "Albert" in response
    )
