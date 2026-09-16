# main.py

from llm_agent.core_v2 import LLMAgent
from PIL import Image, ImageDraw
import tempfile
import os


def create_test_image() -> str:
    """Создаёт тестовое изображение с текстом и возвращает путь."""
    tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    img_path = tmp.name
    tmp.close()

    img = Image.new('RGB', (800, 200), color='white')
    d = ImageDraw.Draw(img)
    d.text((50, 80), "HELLO1WORLD1", fill='black')
    img.save(img_path)

    return img_path


def main():
    """Основная функция для запуска агента."""
    print("Простой LLM-агент с инструментами ('Калькулятор', 'Поиск', 'PDF', 'OCR')")
    print("-" * 70)

    agent = LLMAgent(local=True, ollama_model="llama3.2:3b")

    # =================================================================
    # Сценарий 1: OCR (image_to_text)
    # =================================================================
    print("\n" + "=" * 70)
    print("СЦЕНАРИЙ 1: OCR (image_to_text)")
    print("=" * 70)

    img_path = create_test_image()
    print(f"Создано тестовое изображение: {img_path}")

    try:
        # Английский запрос — llama3.2 лучше понимает
        query2 = f"Extract text from this image: {img_path}"
        print(f"Запрос: {query2}")
        print("-" * 70)
        response2 = agent.process_query(query2)
        print(f"Ответ: {response2}")
    finally:
        if os.path.exists(img_path):
            os.unlink(img_path)


if __name__ == "__main__":
    main()