import requests

def run_prompt_chain(prompts, model="qwen3:0.6b"):
    results = []
    for i, prompt in enumerate(prompts):
        # Добавляем предыдущий результат в контекст
        if i > 0:
            prompt = f"Контекст: {results[-1]}\n\nЗадача: {prompt}"
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False}
        )
        result = response.json()["response"]
        results.append(result)
        print(f"Шаг {i+1} завершен")
    
    return results

# Пример цепочки
chain = [
    "Сгенерируй 10 идей для стартапа в сфере EdTech",
    "Для каждой идеи из предыдущего шага оцени: 1) Потенциальную аудиторию, 2) Барьеры входа, 3) Примеры существующих аналогов",
    "Выбери 3 наиболее перспективные идеи и детализируй бизнес-модель для каждой"
]

results = run_prompt_chain(chain)
print(results)
