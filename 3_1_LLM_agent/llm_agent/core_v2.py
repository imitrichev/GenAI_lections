import requests
import json
import re
from typing import List, Dict, Optional
from decouple import config
from .tool_calculator import CalculatorTool
from .tool_websearch import WebSearchTool
from .tool_audit_logger import AuditLogger


class LLMAgent:
    """
    LLM-агент, который планирует и выполняет задачи с помощью инструментов.
    Поддерживает как OpenRouter API, так и локальный Ollama.
    """
    def __init__(self, model: str = "tngtech/deepseek-r1t2-chimera", local: bool = False, 
                  ollama_base_url: str = "http://localhost:11434", ollama_model: str = "qwen3.5:0.8b"):
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
        self.tools = {
            "calculator": CalculatorTool(),
            "web_search": WebSearchTool(),
        }
        self.audit_logger = AuditLogger()
        self.conversation_history = []

    def _make_api_request(self, payload: Dict, headers: Optional[Dict] = None) -> Dict:
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
        system_prompt = f"""
        You are a helpful AI planning assistant. Analyze the user's request and decide if you need to use any tools.
        Available tools:
        - **calculator**: For any math-related questions (numbers, calculations). Use it with the full expression.
        - **web_search**: For finding any information about the real world (current events, facts, definitions). Use it with the user's question or a clear search query. USE ONLY RUSSIAN LANGUAGE QUERIES in this tool.
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
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_text, re.DOTALL)
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
                json_match = re.search(r'\{.*"plan".*\}', llm_text, re.DOTALL)
                if json_match:
                    action_plan = json.loads(json_match.group())
                    return action_plan.get("plan", [])
            except:
                pass
            return []

    def _generate_final_response(self, user_query: str) -> str:
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
            final_text = response_data["choices"][0]["message"]["content"]
            return final_text
        except Exception as e:
            return f"Ошибка при генерации финального ответа. Детали: {e}"

    def process_query(self, query: str) -> str:
        self.audit_logger.log_request(query)
        print(f"Агент анализирует ваш запрос... (Режим: {'локальный Ollama' if self.local else 'OpenRouter'})")
        plan = self._ask_llm_for_plan(query)
        self.audit_logger.log_plan(plan)
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
                final_response = response_data["choices"][0]["message"]["content"]
                self.audit_logger.log_final_response(final_response)
                return final_response
            except Exception as e:
                error_msg = f"Извините, не удалось сгенерировать ответ: {e}"
                self.audit_logger.log_error(error_msg, "API_ERROR")
                return error_msg
        print(f"План действий: {plan}")
        for step in plan:
            tool_name = step.get('action')
            tool_input = step.get('input')
            if tool_name in self.tools:
                print(f"Выполняется инструмент: '{tool_name}'")
                result = self.tools[tool_name].use(tool_input)
                print(f"Результат: {result}...")
                self.audit_logger.log_tool_execution(tool_name, tool_input, result)
                self.conversation_history.append({
                    'role': 'system',
                    'content': f"Tool {tool_name} result: {result}"
                })
            else:
                error_msg = f"Ошибка: инструмент с именем '{tool_name}' не найден."
                print(error_msg)
                self.audit_logger.log_error(error_msg, "TOOL_NOT_FOUND")
                self.conversation_history.append({'role': 'system', 'content': error_msg})
        print("Составляю финальный ответ...")
        final_response = self._generate_final_response(query)
        self.audit_logger.log_final_response(final_response)
        return final_response

    def test_ollama_connection(self) -> bool:
        if not self.local:
            return False
        try:
            test_url = f"{self.ollama_base_url}/v1/models"
            response = requests.get(test_url)
            return response.status_code == 200
        except:
            return False