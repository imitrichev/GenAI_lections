# llm_agent/core.py

import json
from dataclasses import dataclass
from typing import List, Dict, Optional

import requests

from tools.tool_calculator import CalculatorTool
from tools.tool_spell_checker import SpellCheckerTool
from tools.tool_websearch import WebSearchTool


# from decouple import config

@dataclass
class ProviderConfig:
    """
    Конфигурация LLM-провайдера (OpenRouter или Ollama).
    Вся логика выбора URL, модели и заголовков — здесь.
    """
    url: str
    model: str
    api_key: Optional[str] = None
    stream: bool = False

    @classmethod
    def openrouter(cls, model: str, api_key: Optional[str] = None) -> "ProviderConfig":
        return cls(
            url="https://openrouter.ai/api/v1/chat/completions",
            model=model,
            api_key=api_key,
            stream=False,
        )

    @classmethod
    def ollama(cls, base_url: str, model: str) -> "ProviderConfig":
        return cls(
            url=f"{base_url}/v1/chat/completions",
            model=model,
            api_key=None,
            stream=True,
        )

    @property
    def is_local(self) -> bool:
        return self.api_key is None


class LLMAgent:
    """
    LLM-агент, который планирует и выполняет задачи с помощью инструментов.
    Поддерживает как OpenRouter API, так и локальный Ollama.
    """

    def __init__(self, provider: ProviderConfig):
        """
        Инициализирует агента.
        
        Args:
            provider (ProviderConfig): Конфигурация LLM-провайдера.
        """
        self.provider = provider
        
        # Создаем экземпляры инструментов
        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
            "spell_check": SpellCheckerTool(),
        }
        self.conversation_history = []
    
    def _make_api_request(self, payload: Dict, headers: Optional[Dict] = None) -> Dict:
        """
        Универсальный метод для отправки запросов к API.
        Поддерживает как OpenRouter, так и Ollama.
        
        Args:
            payload (Dict): Тело запроса.
            headers (Dict, optional): Заголовки запроса.
            
        Returns:
            Dict: Ответ от API.
        """
        if headers is None:
            headers = {}
        
        headers["Content-Type"] = "application/json"
        if self.provider.api_key:
            headers["Authorization"] = f"Bearer {self.provider.api_key}"
        
        try:
            response = requests.post(self.provider.url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка при запросе к API: {e}")
    
    def _ask_llm_for_plan(self, query: str) -> List[Dict]:
        """
        Создает план действий, используя LLM.
        Работает как с OpenRouter, так и с Ollama.
        """
        # Системный промпт, который объясняет агенту его роль и формат ответа
        system_prompt = f"""
        You are a helpful AI planning assistant. Analyze the user's request and decide if you need to use any tools.

        Available tools:
        - **calculator**: For any math-related questions (numbers, calculations). Use it with the full expression.
        - **web_search**: For finding any information about the real world (current events, facts, definitions). Use it with the user's question or a clear search query. USE ONLY RUSSIAN LANGUAGE QUERIES in this tool.
        - **spell_check**: For checking spelling of text. Use it with the text that needs spell checking. Supports Russian and English languages.

        Your response MUST be ONLY a JSON object of the following format.
        If one or more tools are needed to answer, return JSON of this structure:
        {{
        "plan": [
            {{"action": "tool_name", "input": "some text to pass into tool"}},
            ... //MORE ACTIONS IF NEEDED SEVERAL TOOLS. ONE ACTION FOR ONE TOOL CALL
        ]
        }}
        If no tool is needed, return an empty plan: {{"plan": []}}.
        """

        # Формируем запрос к API
        payload = {
            "model": self.provider.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        }
        
        try:
            if self.provider.stream:
                payload["stream"] = False
            
            response_data = self._make_api_request(payload)
            
            # Извлекаем текстовый ответ от модели
            llm_text = response_data["choices"][0]["message"]["content"]

            # Очищаем ответ от блоков кода Markdown
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_text, re.DOTALL)
            
            if json_match:
                cleaned_json_text = json_match.group(1)
            else:
                cleaned_json_text = llm_text

            print(f"> Ответ LLM для плана (очищенный): {cleaned_json_text}")
            
            # Пытаемся преобразовать ответ в JSON
            action_plan = json.loads(cleaned_json_text)
            plan = action_plan.get("plan", [])
            return plan
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Произошла ошибка при создании плана: {e}")
            # Пробуем альтернативный подход: извлечь JSON из текста
            try:
                # Ищем JSON в тексте без маркеров
                import re
                json_match = re.search(r'\{.*"plan".*\}', llm_text, re.DOTALL)
                if json_match:
                    action_plan = json.loads(json_match.group())
                    return action_plan.get("plan", [])
            except:
                pass
            return []

    def _generate_final_response(self, user_query: str) -> str:
        """
        Генерирует финальный ответ на основе истории выполнения.
        """
        prompt = f"""
        Based on the following conversation log, provide a direct and helpful answer to the user's original question.
        Be concise and use the information from the tool results to support your answer.

        Original User Question: {user_query}

        Conversation Log:
        {chr(10).join([msg['content'] for msg in self.conversation_history])}
        """
        
        payload = {
            "model": self.provider.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if self.provider.stream:
            payload["stream"] = False
        
        try:
            response_data = self._make_api_request(payload)
            final_text = response_data["choices"][0]["message"]["content"]
            return final_text
        except Exception as e:
            return f"Ошибка при генерации финального ответа. Детали: {e}"

    def process_query(self, query: str) -> str:
        """
        Основной метод для обработки запроса пользователя.
        """
        print(f"Агент анализирует ваш запрос... (Режим: {'локальный Ollama' if self.provider.is_local else 'OpenRouter'})")
        
        # --- Шаг 1: Планирование ---
        plan = self._ask_llm_for_plan(query)

        if not plan:
            print("Инструменты не требуются. Генерирую ответ напрямую.")
            # Генерируем прямой ответ через LLM
            direct_prompt = f"Ответьте на следующий вопрос кратко и информативно: {query}"
            payload = {
                "model": self.provider.model,
                "messages": [{"role": "user", "content": direct_prompt}]
            }
            if self.provider.stream:
                payload["stream"] = False
            try:
                response_data = self._make_api_request(payload)
                return response_data["choices"][0]["message"]["content"]
            except:
                return "Извините, не удалось сгенерировать ответ."

        # --- Шаг 2: Исполнение плана ---
        print(f"План действий: {plan}")
        for step in plan:
            tool_name = step.get('action')
            tool_input = step.get('input')

            if tool_name in self.tools:
                print(f"Выполняется инструмент: '{tool_name}'")
                result = self.tools[tool_name].use(tool_input)
                print(f"Результат: {result}...")
                
                # Добавляем результат в историю
                self.conversation_history.append({
                    'role': 'system',
                    'content': f"Tool {tool_name} result: {result}"
                })
            else:
                error_msg = f"Ошибка: инструмент с именем '{tool_name}' не найден."
                print(error_msg)
                self.conversation_history.append({'role': 'system', 'content': error_msg})
        
        # --- Шаг 3: Генерация финального ответа ---
        print("Составляю финальный ответ...")
        final_response = self._generate_final_response(query)
        return final_response

    def test_ollama_connection(self) -> bool:
        """
        Тестирует соединение с локальным Ollama сервером.
        
        Returns:
            bool: True если соединение успешно, иначе False.
        """
        if not self.provider.is_local:
            return False
        
        try:
            base_url = self.provider.url.rsplit("/v1/chat/completions", 1)[0]
            test_url = f"{base_url}/v1/models"
            response = requests.get(test_url)
            return response.status_code == 200
        except:
            return False