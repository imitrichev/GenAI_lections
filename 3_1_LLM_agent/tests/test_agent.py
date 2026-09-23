import os

import pytest
#from unittest.mock import MagicMock, patch
from llm_agent.core_v2 import LLMAgent

# =====================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ (Запускают реальную Ollama / API)
# =====================================================================
# Маркируем как 'integration', чтобы их можно было отключать при быстрой проверке

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("RUN_INTEGRATION_TESTS") != "1",
        reason="Set RUN_INTEGRATION_TESTS=1 to run tests that require Ollama and internet access.",
    ),
]


def test_calculator_query_live():
    """Реальный запуск агента для проверки математики."""
    # Для тестов лучше использовать локальную модель, если она поднята
    agent = LLMAgent(
        local=True,
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3.5:2b"),
    )
    query = "Сколько будет (5 + 3) * 2? Напиши только цифру."
    
    response = agent.process_query(query)
    
    # Проверяем, что агент смог посчитать и выдать 16
    assert "16" in response


def test_football_query_live():
    """Реальный запуск агента для проверки поиска DuckDuckGo."""
    agent = LLMAgent(
        local=True,
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3.5:2b"),
    )
    query = "Кто выиграл последний матч Спартак-Динамо?"
    
    response = agent.process_query(query)
    
    # Проверяем, что в реальном ответе фигурируют названия команд
    assert "Спартак" in response or "Spartak" in response
    assert "Динамо" in response or "Dynamo" in response
