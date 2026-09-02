# llm_agent/core.py (или core_v2.py)

import requests
import json
import re
from typing import List, Dict, Optional
from decouple import config

from .tool_calculator import CalculatorTool
from .tool_websearch import WebSearchTool
from .tool_geocoding import GeocodingTool

class LLMAgent:
    """
    LLM-агент, который планирует и выполняет задачи с помощью инструментов.
    Поддерживает как OpenRouter API, так и локальный Ollama.
    """

    def __init__(self, model: str = "tngtech/deepseek-r1t2-chimera", local: bool = False, 
                 ollama_base_url: str = "http://localhost:11434", ollama_model: str = "qwen2.5:3b"):
        """
        Инициализирует агента.
        """
        self.local = local
        self.ollama_base_url = ollama_base_url
        self.ollama_model = ollama_model
        
        if not self.local:
            self.api_key = config('OPENROUTER_API_KEY')
            self.url = "https://openrouter.ai/api/v1/chat/completions"
            self.model = model
        else:
            self.api_key = None
            self.url = f"{self.ollama_base_url}/v1/chat/completions"
            self.model = ollama_model
        
        # Создаем экземпляры инструментов
        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
            "geocoding": GeocodingTool(),
        }
        self.conversation_history = []
    
    def _make_api_request(self, payload: Dict, headers: Optional[Dict] = None) -> Dict:
        """
        Универсальный метод для отправки запросов к API.
        """
        if headers is None:
            headers = {}
        
        if not self.local:
            headers.update({
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            })
        else:
            headers["Content-Type"] = "application/json"
        
        try:
            response = requests.post(self.url, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ошибка при запросе к API: {e}")
    
    def _ask_llm_for_plan(self, query: str) -> List[Dict]:
        """
        Создает план действий, используя LLM.
        """
        system_prompt = f"""
        You are a planning assistant. You MUST use the geocoding tool for ANY question about coordinates, cities, or locations.

        Available tools:
        - calculator: For math only.
        - web_search: For news and facts.
        - geocoding: For coordinates and cities. Format: "coords: City" or "address: lat, lon".

        CRITICAL: If user asks about coordinates or city names, you MUST use geocoding tool.

        Example:
        User: "Какие координаты у Москвы?"
        You: {{"plan": [{{"action": "geocoding", "input": "coords: Москва"}}]}}

        Return ONLY valid JSON. Do not include any explanations or <think> tags.
        If one or more tools are needed, return JSON of this structure:
        {{
          "plan": [
            {{"action": "tool_name", "input": "some text to pass into tool"}}
          ]
        }}
        If no tool is needed, return: {{"plan": []}}.
        """

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        }
        
        try:
            if self.local:
                payload["stream"] = False
            
            response_data = self._make_api_request(payload)
            llm_text = response_data["choices"][0]["message"]["content"]

            # 1. КРИТИЧЕСКИ ВАЖНО: Удаляем теги рассуждений <think>...</think> от моделей Qwen
            llm_text = re.sub(r'<think>.*?</think>', '', llm_text, flags=re.DOTALL).strip()

            # 2. Ищем JSON в markdown блоках (используем жадный захват .*, а не .*?)
            json_match = re.search(r'```(?:json)?\s*(\{.*\})\s*```', llm_text, re.DOTALL)
            
            if json_match:
                cleaned_json_text = json_match.group(1)
            else:
                # 3. Запасной вариант: ищем просто первую { и последнюю } во всем тексте
                json_match = re.search(r'(\{.*\})', llm_text, re.DOTALL)
                cleaned_json_text = json_match.group(1) if json_match else llm_text

            print(f"> Ответ LLM для плана (очищенный): {cleaned_json_text}")
            
            # Пытаемся преобразовать ответ в JSON
            action_plan = json.loads(cleaned_json_text)
            return action_plan.get("plan", [])
            
        except (json.JSONDecodeError, KeyError, Exception) as e:
            print(f"Произошла ошибка при создании плана: {e}")
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
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        if self.local:
            payload["stream"] = False
        
        try:
            response_data = self._make_api_request(payload)
            return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Ошибка при генерации финального ответа. Детали: {e}"

    def process_query(self, query: str) -> str:
        """
        Основной метод для обработки запроса пользователя.
        """
        print(f"Агент анализирует ваш запрос... (Режим: {'локальный Ollama' if self.local else 'OpenRouter'})")
        
        # --- Шаг 1: Планирование ---
        plan = self._ask_llm_for_plan(query)

        if not plan:
            print("Инструменты не требуются. Генерирую ответ напрямую.")
            direct_prompt = f"Ответьте на следующий вопрос кратко и информативно: {query}"
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": direct_prompt}]
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
            tool_name = step.get('action')
            tool_input = step.get('input')

            if tool_name in self.tools:
                print(f"Выполняется инструмент: '{tool_name}'")
                result = self.tools[tool_name].use(tool_input)
                print(f"Результат: {result}")
                
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
        return self._generate_final_response(query)

    def test_ollama_connection(self) -> bool:
        """
        Тестирует соединение с локальным Ollama сервером.
        """
        if not self.local:
            return False
        
        try:
            test_url = f"{self.ollama_base_url}/v1/models"
            response = requests.get(test_url)
            return response.status_code == 200
        except:
            return False