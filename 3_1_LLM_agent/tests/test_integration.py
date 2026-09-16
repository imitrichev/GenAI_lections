import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../llm_agent')))

from fallback_model import FallbackModel

# Проверка доступности Ollama
def is_ollama_available():
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False

ollama_available = is_ollama_available()

def test_ollama_only():
    """
    Тест 1: Работа только с Ollama (без OpenRouter)
    """
    print("=" * 60)
    print("ТЕСТ 1: Только Ollama (OpenRouter отключён)")
    print("=" * 60)
    
    # Создаём модель БЕЗ ключа OpenRouter → сразу идём в Ollama
    model = FallbackModel(
        openrouter_api_key=None,  # None = не использовать OpenRouter
        ollama_model="qwen2.5:0.5b"
    )
    
    messages = [
        {"role": "system", "content": "Ты полезный ассистент. Отвечай кратко."},
        {"role": "user", "content": "Сколько будет 2 + 2?"}
    ]
    
    print(f"Запрос: {messages[-1]['content']}")
    print("Отправка запроса в Ollama...")
    
    try:
        response = model.generate(messages)
        answer = response["choices"][0]["message"]["content"]
        print(f"\nОтвет от Ollama:\n{answer}")
    except Exception as e:
        print(f"\nОшибка: {e}")
        print("💡 Убедитесь, что Ollama запущена (ollama serve)")
    
    print()


def test_fallback():
    """
    Тест 2: Fallback с OpenRouter на Ollama
    """
    print("=" * 60)
    print("ТЕСТ 2: Fallback (OpenRouter → Ollama)")
    print("=" * 60)
    print("Примечание: OpenRouter упадёт (неверный ключ),")
    print("и код автоматически переключится на Ollama\n")
    
    # Создаём модель с НЕВЕРНЫМ ключом → OpenRouter упадёт → fallback на Ollama
    model = FallbackModel(
        openrouter_api_key="invalid_key",  # Неверный ключ → ошибка
        openrouter_model="test-model",
        ollama_model="qwen2.5:0.5b"
    )
    
    messages = [
        {"role": "user", "content": "Привет! Напиши короткое стихотворение."}
    ]
    
    print(f"Запрос: {messages[0]['content']}")
    print("Попытка отправки в OpenRouter...")
    
    try:
        response = model.generate(messages)
        answer = response["choices"][0]["message"]["content"]
        print(f"\nОтвет (от Ollama, после fallback):\n{answer}")
    except Exception as e:
        print(f"\nКритическая ошибка (оба API недоступны): {e}")
    
    print()


def test_both_apis_down():
    """
    Тест 3: Оба API недоступны
    """
    print("=" * 60)
    print("ТЕСТ 3: Оба API недоступны")
    print("=" * 60)
    print("Останавливаем Ollama (Ctrl+C в терминале с ollama serve)")
    print("и запускаем этот тест → должна быть ошибка\n")
    
    model = FallbackModel(
        openrouter_api_key=None,  # Не используем OpenRouter
        ollama_base_url="http://localhost:9999",  # Несуществующий порт
        ollama_model="qwen2.5:0.5b"
    )
    
    messages = [{"role": "user", "content": "Тест"}]
    
    try:
        response = model.generate(messages)
        print("Неожиданно: запрос прошёл успешно")
    except Exception as e:
        print(f"Ожидаемая ошибка: {e}")
    
    print()


if __name__ == "__main__":
    print("\nТЕСТИРОВАНИЕ FALLBACKMODEL (Вариант 14)\n")
    
    test_ollama_only()
    test_fallback()
    test_both_apis_down()