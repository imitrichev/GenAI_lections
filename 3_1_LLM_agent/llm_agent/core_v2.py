# llm_agent/core.py

import json
import re
from typing import Dict, List, Optional
from decouple import config
import requests

from .tool_calculator import CalculatorTool
from .tool_excelreader import ExcelReaderTool  # <-- 1. Импортируем новый tool
from .tool_websearch import WebSearchTool


class LLMAgent:
    """LLM-агент, который планирует и выполняет задачи с помощью инструментов.

    Поддерживает как OpenRouter API, так и локальный Ollama.
    """

    def __init__(
        self,
        model: str = "tngtech/deepseek-r1t2-chimera",
        local: bool = False,
        ollama_base_url: str = "http://localhost:11434",
        ollama_model: str = "qwen3:0.6b",
    ):
        """Инициализирует агента."""
        self.local = local
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model

        if not self.local:
            self.api_key = config("OPENROUTER_API_KEY")
            self.url = "https://openrouter.ai/api/v1/chat/completions"
            self.model = model
        else:
            self.api_key = None
            self.url = f"{self.ollama_base_url}/v1/chat/completions"
            self.model = ollama_model

        # Создаем экземпляры инструментов
        # <-- 2. Регистрируем excel_reader в словаре инструментов
        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
            "excel_reader": ExcelReaderTool(),
        }
        self.conversation_history = []

    def _make_api_request(
        self, payload: Dict, headers: Optional[Dict] = None
    ) -> Dict:
        """Универсальный метод для отправки запросов к API.

        Поддерживает как OpenRouter, так и Ollama.
        """
        if headers is None:
            headers = {}

        if not self.local:
            headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
        else:
            headers["Content-Type"] = "application/json"

        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка при запросе к API: {e}")

    def _ask_llm_for_plan(self, query: str) -> List[Dict]:
        """Создает план действий, используя LLM.

        Работает как с OpenRouter, так и с Ollama.
        """
        # <-- 3. Обновляем system_prompt: добавляем описание excel_reader
        system_prompt = f"""
        You are a helpful AI planning assistant. Analyze the user's request and decide if you need to use any tools.

        Available tools:
        - **calculator**: For any math-related questions (numbers, calculations). Use it with the full expression.
        - **web_search**: For finding any information about the real world (current events, facts, definitions). Use it with the user's question or a clear search query. USE ONLY RUSSIAN LANGUAGE QUERIES in this tool.
        - **excel_reader**: For reading data from Excel files (.xlsx, .xls) via file path or URL. Pass the file path or URL as the input.

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
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        }

        try:
            if self.local:
                payload["stream"] = False

            response_data = self._make_api_request(payload)
            llm_text = response_data["choices"][0]["message"]["content"]

            json_match = re.search(
                r"```(?:json)?\s*(\{.*?\})\s*```", llm_text, re.DOTALL
            )

            if json_match:
                cleaned_json_text = json_match.group(1)
            else:
                cleaned_json_text = llm_text

            print(f"> Ответ LLM для плана (очищенный): {cleaned_json_text}")

            action_plan = json.loads(cleaned_json_text)
            plan = action_plan.get("plan", [])
            return plan

        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Произошла ошибка при создании плана: {e}")
            try:
                json_match = re.search(r"\{.*\"plan\".*\}", llm_text, re.DOTALL)
                if json_match:
                    action_plan = json.loads(json_match.group())
                    return action_plan.get("plan", [])
            except:
                pass
            return []

    def _generate_final_response(self, user_query: str) -> str:
        """Генерирует финальный ответ на основе истории выполнения."""
        prompt = f"""
        Based on the following conversation log, provide a direct and helpful answer to the user's original question.
        Be concise and use the information from the tool results to support your answer.

        Original User Question: {user_query}

        Conversation Log:
        {chr(10).join([msg['content'] for msg in self.conversation_history])}
        """

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }

        if self.local:
            payload["stream"] = False

        try:
            response_data = self._make_api_request(payload)
            final_text = response_data["choices"][0]["message"]["content"]
            return final_text
        except Exception as e:
            return f"Ошибка при генерации финального ответа. Детали: {e}"

    def process_query(self, query: str) -> str:
        """Основной метод для обработки запроса пользователя."""
        print(
            f"Агент анализирует ваш запрос... (Режим: {'локальный Ollama' if self.local else 'OpenRouter'})"
        )

        # --- Шаг 1: Планирование ---
        plan = self._ask_llm_for_plan(query)

        if not plan:
            print("Инструменты не требуются. Генерирую ответ напрямую.")
            direct_prompt = (
                f"Ответьте на следующий вопрос кратко и информативно: {query}"
            )
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": direct_prompt}],
            }
            if self.local:
                payload["stream"] = False
            try:
                response_data = self._make_api_request(payload)
                return response_data["choices"][0]["message"]["content"]
            except:
                return "Извините, не удалось сгенерировать ответ."

        # --- Шаг 2: Исполнение плана ---
        print(f"План действий: {plan}")
        for step in plan:
            tool_name = step.get("action")
            tool_input = step.get("input")

            if tool_name in self.tools:
                print(f"Выполняется инструмент: '{tool_name}'")
                result = self.tools[tool_name].use(tool_input)
                print(f"Результат: {result}...")

                self.conversation_history.append(
                    {
                        "role": "system",
                        "content": f"Tool {tool_name} result: {result}",
                    }
                )
            else:
                error_msg = (
                    f"Ошибка: инструмент с именем '{tool_name}' не найден."
                )
                print(error_msg)
                self.conversation_history.append(
                    {"role": "system", "content": error_msg}
                )

        # --- Шаг 3: Генерация финального ответа ---
        print("Составляю финальный ответ...")
        final_response = self._generate_final_response(query)
        return final_response

    def test_ollama_connection(self) -> bool:
        """Тестирует соединение с локальным Ollama сервером."""
        if not self.local:
            return False

        try:
            test_url = f"{self.ollama_base_url}/v1/models"
            response = requests.get(test_url)
            return response.status_code == 200
        except:
            return False
