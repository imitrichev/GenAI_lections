from llm_agent.core_v2 import LLMAgent

print(" Инициализация агента в локальном режиме (Ollama)...")
# local=True означает, что агент будет стучаться в твою установленную Ollama, а не в интернет
agent = LLMAgent(local=True, ollama_model="qwen3:0.6b") 

# Задаем вопрос, который заставит агента использовать калькулятор
query = "Сколько будет 125 умножить на 4?"

print(f"\n Запрос пользователя: {query}")
print("-" * 40)

# Запускаем процесс
result = agent.process_query(query)

print("-" * 40)
print(f" Финальный ответ агента: {result}")
print("\n Готово! Теперь проверь, появился ли файл audit_logs.json в этой папке.")
