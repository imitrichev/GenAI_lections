# main.py

from llm_agent.core_v2 import LLMAgent
import os


def main():
    """Основная функция для запуска агента."""
    print("Простой LLM-агент с инструментами ('Калькулятор', 'Поиск в DuckDuckGo', 'QR-декодер')")
    print("-" * 70)

    # Создаем агента в локальном режиме (Ollama)
    agent = LLMAgent(local=True, ollama_model="qwen3:0.6b")

    # Создаем тестовый QR-код, если его нет
    if not os.path.exists("test_qr.png"):
        print("Создаю тестовый QR-код...")
        import qrcode
        img = qrcode.make("Hello, World! This is a test QR code for lab work.")
        img.save("test_qr.png")
        print("Тестовый QR-код сохранен в test_qr.png")

    # 👇 ЗДЕСЬ ВАШ ЗАПРОС ДЛЯ QR-ДЕКОДЕРА
    query = "Декодируй QR-код из файла test_qr.png"

    print(f"Ваш запрос: {query}")
    print("-" * 70)

    response = agent.process_query(query)

    print("\n" + "=" * 70)
    print("Финальный ответ агента:\n")
    print(response)
    print("=" * 70)


if __name__ == "__main__":
    main()