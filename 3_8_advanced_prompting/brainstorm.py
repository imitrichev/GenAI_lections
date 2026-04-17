import requests
from decouple import config

class BrainstormClusterPrioritizeChain:
    
    def __init__(self, llm, max_ideas=20):
        """
        Инициализация цепи мозгового штурма
        
        Args:
            llm: объект LLM с методом query()
            max_ideas: максимальное количество идей за генерацию
        """
        self.llm = llm
        self.max_ideas = max_ideas
    
    def execute(self, topic):
        """
        Выполнение полного цикла: генерация -> кластеризация -> приоритизация
        
        Args:
            topic: тема для мозгового штурма
            
        Returns:
            dict: результаты всех фаз
        """
        if not self.llm:
            raise ValueError("LLM не инициализирован. Передайте llm в конструктор.")
        # Фаза 1: Генерация идей
        brainstorm_prompt = f"""
        Генерация идей по теме: {topic}
        
        Правила мозгового штурма:
        1. Количество важнее качества
        2. Не критиковать идеи
        3. Комбинировать и улучшать
        4. Думать нестандартно
        
        Сгенерируй {self.max_ideas} идей. Каждая - в новой строке.
        """
        
        ideas = self.llm.query(brainstorm_prompt)
        
        # Фаза 2: Кластеризация
        cluster_prompt = f"""
        Сгруппируй идеи по тематическим кластерам:
        
        Идеи: {ideas}
        
        Создай 5-7 кластеров с названиями.
        Для каждого кластера укажи:
        - Название
        - Ключевые идеи
        - Общую тему
        """
        
        clusters = self.llm.query(cluster_prompt)
        
        # Фаза 3: Приоритизация
        prioritize_prompt = f"""
        Оцени кластеры по критериям:
        
        Кластеры: {clusters}
        
        Критерии оценки:
        1. Инновационность (1-10)
        2. Практическая ценность (1-10)
        3. Сложность реализации (1-10, где 1 - легко)
        4. Потенциальный эффект (1-10)
        
        Ранжируй кластеры по приоритетности.
        """
        
        prioritized = self.llm.query(prioritize_prompt)
        
        return {
            "ideas": ideas,
            "clusters": clusters,
            "prioritized": prioritized
        }

# Пример использования
class MockLLM:
    """Заглушка для тестирования"""
    def query(self, prompt):
        return """- Создание катализатора на основе наночастиц
        - Использование ИИ для оптимизации процессов
        - Биокатализ для очистки стоков"""

class ollamaLLM:
    def query(self, prompt):
        """Запрос к LLM"""
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "qwen3:4b", "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "")

        
from decouple import config

# Конфигурация
class Config:  
    # OpenRouter
    OPENROUTER_KEY = config('OPENROUTER_API_KEY')
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_MODEL = "qwen/qwen3-coder-next"

class openrouterLLM:
    def query(self, prompt, conversation_history=None):
        """Запрос к OpenRouter с поддержкой контекста"""
        try:
            headers = {
                "Authorization": f"Bearer {Config.OPENROUTER_KEY}",
                "Content-Type": "application/json"
            }
            
            # Формируем историю сообщений
            messages = []
            if conversation_history:
                for i, msg in enumerate(conversation_history[-5:]):  # Последние 5
                    role = "user" if i % 2 == 0 else "assistant"
                    messages.append({"role": role, "content": msg})
            
            messages.append({"role": "user", "content": prompt})
            
            data = {
                "model": Config.OPENROUTER_MODEL,
                "messages": messages,
                "temperature": 0.3
            }
            
            response = requests.post(
                Config.OPENROUTER_URL,
                headers=headers,
                json=data,
                timeout=60
            )
            
            if response.status_code == 200:
                print(response.json()["choices"][0]["message"]["content"])
                print(response.status_code)
                return response.json()["choices"][0]["message"]["content"]
            else:
                print(response.status_code)
            return ""
        except:
            print(response.status_code)
            return ""

# Использование
llm = openrouterLLM() #ollamaLLM() #MockLLM()
chain = BrainstormClusterPrioritizeChain(llm, max_ideas=15)
result = chain.execute("Новые катализаторы для нефтепереработки")

print("\n=== РЕЗУЛЬТАТ ===")
res=len(list(filter(None, result['ideas'].split('\n'))))
print(f"Сгенерировано идей: {res}")
print(f"Кластеры: {result['clusters']}")
print(f"Приоритет: {result['prioritized']}")