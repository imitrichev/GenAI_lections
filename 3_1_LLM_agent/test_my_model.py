import pytest
import sys
import os

# ИСПРАВЛЕНИЕ: убираем '../', так как llm_agent лежит в той же папке, что и этот скрипт
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(current_dir, 'llm_agent'))

from fallback_model import FallbackModel
import requests

# Проверка доступности Ollama
def is_ollama_available():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except:
        return False

ollama_available = is_ollama_available()

def test_ollama_only():
    print("=" * 60)
    print("ТЕСТ 1: Только Ollama (OpenRouter отключён)")
    print("=" * 60)
    
    model = FallbackModel(
        openrouter_api_key=None,
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
        print(f"\n✅ Ответ от Ollama:\n{answer}")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        print("💡 Убедитесь, что Ollama запущена (ollama serve)")
    print()


def test_fallback():
    print("=" * 60)
    print("ТЕСТ 2: Fallback (OpenRouter → Ollama)")
    print("=" * 60)
    print("Примечание: OpenRouter упадёт (неверный ключ),")
    print("и код автоматически переключится на Ollama\n")
    
    model = FallbackModel(
        openrouter_api_key="invalid_key",
        openrouter_model="test-model",
        ollama_model="qwen2.5:0.5b"
    )
    
    messages = [{"role": "user", "content": "Привет! Напиши короткое стихотворение."}]
    
    print(f"Запрос: {messages[0]['content']}")
    print("Попытка отправки в OpenRouter...")
    
    try:
        response = model.generate(messages)
        answer = response["choices"][0]["message"]["content"]
        print(f"\n✅ Ответ (от Ollama, после fallback):\n{answer}")
    except Exception as e:
        print(f"\n❌ Критическая ошибка (оба API недоступны): {e}")
    print()


def test_both_apis_down():
    print("=" * 60)
    print("ТЕСТ 3: Оба API недоступны")
    print("=" * 60)
    
    model = FallbackModel(
        openrouter_api_key=None,
        ollama_base_url="http://localhost:9999",  # Несуществующий порт
        ollama_model="qwen2.5:0.5b"
    )
    
    messages = [{"role": "user", "content": "Тест"}]
    
    try:
        response = model.generate(messages)
        print("❌ Неожиданно: запрос прошёл успешно")
    except Exception as e:
        print(f"✅ Ожидаемая ошибка: {e}")
    print()


if __name__ == "__main__":
    print("\n🧪 ТЕСТИРОВАНИЕ FALLBACKMODEL (Вариант 14)\n")
    
    test_ollama_only()
    
    # Тест 2 запускаем только если Ollama доступна (иначе некуда будет переключаться)
    if ollama_available:
        test_fallback()
    else:
        print("⚠️ Тест 2 пропущен: Ollama недоступна\n")
        
    test_both_apis_down()
    
    print("=" * 60)
    print("✅ Тестирование завершено!")
    print("=" * 60)