"""
Упрощенный конвейер для извлечения данных о перовскитах из лабораторных журналов
Извлекаем: формула перовскита, температура спекания, время спекания
"""

import os
import requests
import pandas as pd
from pathlib import Path
from PIL import Image
import pytesseract
import json
from decouple import config

# Конфигурация
class Config:
    API_CHOICE = "ollama"  # "ollama" или "openrouter"
    
    # Ollama
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "qwen3:0.6b"
    
    # OpenRouter
    OPENROUTER_KEY = config('OPENROUTER_API_KEY')
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_MODEL = "qwen/qwen3-next-80b-a3b-instruct:free"
    
    # Папки
    IMAGES_FOLDER = "./perovskite_journals"
    OUTPUT_FILE = "perovskite_data.xlsx"

# Функции для работы с LLM
def ask_ollama(prompt, conversation_history=None):
    """Запрос к Ollama с поддержкой контекста"""
    try:
        # Формируем полный промпт с историей
        full_prompt = ""
        if conversation_history:
            for msg in conversation_history[-5:]:  # Берем последние 5 сообщений
                full_prompt += f"{msg}\n\n"
        
        full_prompt += prompt
        
        response = requests.post(
            Config.OLLAMA_URL,
            json={
                "model": Config.OLLAMA_MODEL,
                "prompt": full_prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json().get("response", "")
        return f"Ошибка: {response.status_code}"
    except:
        return ""

def ask_openrouter(prompt, conversation_history=None):
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
            return response.json()["choices"][0]["message"]["content"]
        return ""
    except:
        return ""

def ask_llm(prompt, conversation_history=None):
    """Универсальная функция запроса"""
    if Config.API_CHOICE == "ollama":
        return ask_ollama(prompt, conversation_history)
    else:
        return ask_openrouter(prompt, conversation_history)

# Основной класс
class PerovskiteJournalProcessor:
    """Обработчик журналов для извлечения данных о перовскитах"""
    
    def __init__(self):
        self.conversation_history = []
    
    def extract_text(self, image_path):
        """Извлечение текста из изображения с помощью OCR"""
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img, lang='rus+eng')
            return text
        except:
            return ""
    
    def extract_perovskite_data(self, text):
        """Извлечение данных о перовскитах с помощью LLM"""

        print("Обработка текста:")
        print(text)
        
        # Шаг 1: Найти все перовскиты
        prompt1 = f"""
        Из текста лабораторного журнала найди все упоминания перовскитов.
        Перовскиты - это материалы с формулами типа La0.5Sr0.5MnO3, BaTiO3, SrTiO3.
        
        Текст:
        {text}  # Ограничиваем текст для скорости
        
        Найди все химические формулы перовскитов. Верни ТОЛЬКО JSON список формул:
        {{"perovskites": ["формула1", "формула2", ...]}}
        """
        
        self.conversation_history.append(prompt1)
        response1 = ask_llm(prompt1, self.conversation_history)
        self.conversation_history.append(response1)

        print("Ответ модели:")
        print(response1)
        
        # Парсим JSON с перовскитами
        try:
            start = response1.find('{')
            end = response1.rfind('}') + 1
            json_str = response1[start:end]
            perovskites_data = json.loads(json_str)
            perovskites = perovskites_data.get("perovskites", [])
        except:
            perovskites = []
        
        if not perovskites:
            return []
        
        # Шаг 2: Для каждого перовскита найти температуру и время спекания
        results = []
        
        for perovskite in perovskites[:3]:  # Ограничиваем 3 перовскитами для демо
            prompt2 = f"""
            Для ОДНОГО перовскита {perovskite} найди температуру спекания и время спекания.
            
            Текст журнала:
            {text}
            
            Ищи фразы типа:
            - "спекали при 1200°C в течение 2 часов"
            - "температура спекания 1100°C, время 4 ч"
            - "синтезировали при 1000°C 3 часа"
            
            Верни ТОЛЬКО JSON:
            {{
                "perovskite": "{perovskite}",
                "sintering_temperature": "температура в °C",
                "sintering_time": "время в часах"
            }}
            
            Если данных нет, укажи "не указано".
            """
            
            self.conversation_history.append(prompt2)
            response2 = ask_llm(prompt2, self.conversation_history)
            self.conversation_history.append(response2)

            print("Ответ модели:")
            print(response2)
            
            try:
                start = response2.find('{')
                end = response2.rfind('}') + 1
                json_str = response2[start:end]
                data = json.loads(json_str)
                
                # Очищаем данные
                perovskite_formula = data.get("perovskite", "").strip()
                temperature = data.get("sintering_temperature", "не указано").replace('°C', '').strip()
                time = data.get("sintering_time", "не указано").replace('часов', '').replace('ч', '').strip()
                
                if perovskite_formula and temperature != "не указано":
                    results.append({
                        "Формула перовскита": perovskite_formula,
                        "Температура спекания (°C)": temperature,
                        "Время спекания (часы)": time
                    })
            except:
                continue
        
        return results
    
    def process_image(self, image_path):
        """Обработка одного изображения"""
        print(f"\n📄 Обработка: {image_path.name}")
        
        # Извлекаем текст
        text = self.extract_text(image_path)
        if len(text) < 10:
            print("  ⚠️  Мало текста, пропускаем")
            return []
        
        print(f"  ✓ Текст извлечен ({len(text)} символов)")
        
        # Извлекаем данные о перовскитах
        results = self.extract_perovskite_data(text)
        
        if results:
            print(f"  ✓ Найдено {len(results)} перовскитов")
            for r in results:
                print(f"    - {r['Формула перовскита']}: {r['Температура спекания (°C)']}°C, {r['Время спекания (часы)']} ч")
        else:
            print("  ⚠️  Перовскиты не найдены")
        
        return results
    
    def process_folder(self, folder_path):
        """Обработка всех изображений в папке"""
        folder = Path(folder_path)
        
        if not folder.exists():
            print(f"✗ Папка {folder_path} не найдена")
            return pd.DataFrame()
        
        # Ищем изображения
        image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.jpeg")) + list(folder.glob("*.png"))
        
        if not image_files:
            print(f"✗ В папке нет изображений")
            return pd.DataFrame()
        
        print(f"📁 Найдено {len(image_files)} изображений")
        
        # Обрабатываем каждое изображение
        all_results = []
        
        for image_file in image_files:
            # Сбрасываем историю для каждого нового изображения
            self.conversation_history = []
            
            results = self.process_image(image_file)
            all_results.extend(results)
        
        # Сохраняем в DataFrame
        if all_results:
            df = pd.DataFrame(all_results)
            return df
        else:
            print("✗ Данные не найдены")
            return pd.DataFrame()

# Демонстрация с тестовыми данными
def create_test_image():
    """Создание тестового изображения с текстом о перовскитах"""
    from PIL import Image, ImageDraw, ImageFont
    import os
    
    # Создаем простое изображение с текстом
    img = Image.new('RGB', (800, 400), color='white')
    draw = ImageDraw.Draw(img)
    
    # Простой текст без шрифта (для демо)
    text = """
    Лабораторный журнал №45
    Синтез перовскитных материалов
    
    Образец 1: La0.7Sr0.3MnO3
    Спекание: 1200°C в течение 4 часов
    
    Образец 2: Ba0.5Sr0.5TiO3
    Температура спекания: 1100°C
    Время: 2 часа
    
    Образец 3: La0.5Sr0.5Fe0.3Co0.7O3
    Условия спекания: 1150°C, 3 ч
    
    Результаты: все образцы показали хорошую кристалличность.
    """
    
    # Рисуем текст построчно
    y = 20
    for line in text.strip().split('\n'):
        draw.text((20, y), line.strip(), fill='black')
        y += 25
    
    # Сохраняем
    test_folder = Path(Config.IMAGES_FOLDER)
    test_folder.mkdir(exist_ok=True)
    
    test_path = test_folder / "test_perovskite.jpg"
    img.save(test_path)
    print(f"✓ Тестовое изображение создано: {test_path}")
    
    return test_path

# Основная функция
def main():
    print("="*60)
    print("ИЗВЛЕЧЕНИЕ ДАННЫХ О ПЕРОВСКИТАХ ИЗ ЖУРНАЛОВ")
    print("="*60)
    
    # Создаем тестовое изображение если нет реальных данных
    test_folder = Path(Config.IMAGES_FOLDER)
    if not list(test_folder.glob("*.*")):
        print("Создаю тестовое изображение...")
        create_test_image()
    
    # Обрабатываем папку
    processor = PerovskiteJournalProcessor()
    df = processor.process_folder(Config.IMAGES_FOLDER)
    
    # Сохраняем результаты
    if not df.empty:
        df.to_excel(Config.OUTPUT_FILE, index=False)
        print(f"\n✅ Результаты сохранены в {Config.OUTPUT_FILE}")
        
        # Показываем таблицу
        print("\n📊 ИЗВЛЕЧЕННЫЕ ДАННЫЕ:")
        print(df.to_string(index=False))
    else:
        print("\n❌ Данные не извлечены")

if __name__ == "__main__":
    # Проверяем наличие Tesseract
    try:
        pytesseract.get_tesseract_version()
    except:
        print("⚠️  Установите Tesseract OCR:")
        print("  pip install pip")
        print("  И установите сам Tesseract с поддержкой русского https://github.com/UB-Mannheim/tesseract/wiki")
        exit(1)
    
    main()