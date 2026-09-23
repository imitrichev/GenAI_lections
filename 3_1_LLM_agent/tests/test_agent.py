import pytest
#from unittest.mock import MagicMock, patch
from llm_agent.core_v2 import LLMAgent

# =====================================================================
# ИНТЕГРАЦИОННЫЕ ТЕСТЫ (Запускают реальную Ollama / API)
# =====================================================================

@pytest.mark.integration
def test_todo_add_live():
    """Реальный запуск агента: добавление задачи в todo-лист."""
    agent = LLMAgent(local=True, ollama_model="qwen3.5:2b")
    query = "Используй инструмент todo_list и добавь задачу 'Купить хлеб'"

    response = agent.process_query(query)


    assert "добавлен" in response.lower() or "add" in response.lower()


@pytest.mark.integration
def test_todo_list_live():
    """Реальный запуск агента: вывод списка задач."""
    agent = LLMAgent(local=True, ollama_model="qwen3.5:2b")
    query = "Используй инструмент todo_list и покажи список задач"

    response = agent.process_query(query)


    lower = response.lower()
    assert "список" in lower or "задач" in lower or "пуст" in lower


@pytest.mark.integration
def test_todo_delete_live():
    """Реальный запуск агента: удаление задачи по id."""
    agent = LLMAgent(local=True, ollama_model="qwen3.5:2b")


    agent.process_query("Используй инструмент todo_list и добавь задачу 'Тестовая задача'")

    query = "Используй инструмент todo_list и удали задачу с айди 1"
    response = agent.process_query(query)

    lower = response.lower()
    assert "удал" in lower or "delete" in lower


@pytest.mark.integration
def test_todo_mark_live():
    """Реальный запуск агента: toggle отметки задачи по id."""
    agent = LLMAgent(local=True, ollama_model="qwen3.5:2b")

    # Создаём задачу для отметки
    agent.process_query("Используй инструмент todo_list и добавь задачу 'Отметить меня'")

    query = "Используй инструмент todo_list и отметь задачу с айди 1"
    response = agent.process_query(query)

    lower = response.lower()
  
    assert "отмеч" in lower or "выполн" in lower or "mark" in lower


@pytest.mark.integration
def test_todo_multi_step_live():
    """Реальный запуск агента: многошаговый сценарий (delete + mark + list)."""
    agent = LLMAgent(local=True, ollama_model="qwen3.5:2b")

    # Готовим данные: две задачи
    agent.process_query("Используй инструмент todo_list и добавь задачу 'Первая'")
    agent.process_query("Используй инструмент todo_list и добавь задачу 'Вторая'")

    query = (
        "Используй инструмент todo_list и удали оттуда задачу с айди 1. "
        "Потом отметь задачу с айди 2. Потом выведи мне список задач"
    )
    response = agent.process_query(query)

    lower = response.lower()

    assert "список" in lower or "задач" in lower