from llm_agent.core_v2 import LLMAgent
from fallback_model import FallbackModel
from typing import Dict, Optional

class FallbackLLMAgent(LLMAgent):
    """
    Агент, использующий FallbackModel для вызовов LLM.
    Сохраняет все инструменты (калькулятор, поиск) и логику планирования.
    """
    def __init__(self, fallback_model: FallbackModel, **kwargs):
        # Инициализируем родителя с local=True, чтобы не требовать API-ключ
        # и не создавать лишних зависимостей.
        super().__init__(local=True, **kwargs)
        self.fallback_model = fallback_model

    def _make_api_request(self, payload: Dict, headers: Optional[Dict] = None) -> Dict:
        """
        Переопределяем метод, который делает реальный HTTP-запрос.
        Вместо этого вызываем fallback_model.generate() с извлечённым промптом.
        Возвращаем ответ в формате, ожидаемом родительским классом.
        """
        # Извлекаем промпт из payload
        messages = payload.get('messages', [])
        if not messages:
            prompt = ""
        else:
            # Берём последнее сообщение пользователя (или всё содержимое)
            # В _ask_llm_for_plan используется system + user, нам нужен полный контекст.
            # Для простоты объединим все сообщения в один промпт.
            prompt_parts = [msg['content'] for msg in messages]
            prompt = "\n".join(prompt_parts)

        # Вызываем наш FallbackModel
        response_text = self.fallback_model.generate(prompt)
        if response_text is None:
            response_text = "Не удалось получить ответ от модели."

        # Возвращаем структуру, аналогичную ответу OpenRouter/Ollama
        return {
            "choices": [
                {
                    "message": {
                        "content": response_text
                    }
                }
            ]
        }