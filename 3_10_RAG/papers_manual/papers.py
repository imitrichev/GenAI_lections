"""
Упрощённая RAG-система для поиска в PDF документах
Работает с Ollama (локально) или OpenRouter
"""

import os
import re
import requests
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF
from decouple import config

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

class Config:
    # Выбор API: "ollama" или "openrouter"
    API_CHOICE = "ollama"  # поменяйте при необходимости
    
    # Настройки Ollama
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "qwen3:0.6b"
    
    # Настройки OpenRouter (если нужно)
    OPENROUTER_KEY = config('OPENROUTER_API_KEY')
    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
    OPENROUTER_MODEL = "qwen/qwen3-coder-next"
    
    # Пути
    DATA_DIR = "./data/pdf"  # папка с PDF файлами
    CHROMA_DB = "./chroma_db"
    
    # Модель для эмбеддингов (локальная)
    EMBEDDING_MODEL = "multilingual-e5-small" # "all-MiniLM-L6-v2"

# ============================================================================
# РАБОТА С PDF
# ============================================================================

def extract_text_from_pdf(pdf_path):
    """Извлекает текст из PDF файла"""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text.strip()
    except:
        return ""

def chunk_text(text, chunk_size=500, overlap=100):
    """Разбивает текст на чанки"""
    words = text.split()
    if len(words) <= chunk_size:
        return [" ".join(words)]
    
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# ============================================================================
# ВЕКТОРНАЯ БАЗА ДАННЫХ
# ============================================================================

class VectorDB:
    """Простая векторная база данных на ChromaDB"""
    
    def __init__(self):
        # Используем локальную модель для эмбеддингов
        self.embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Создаём/загружаем базу
        self.client = chromadb.PersistentClient(path=Config.CHROMA_DB)
        
        try:
            self.collection = self.client.get_collection("documents", embedding_function=self.embedder)
            print("✓ База данных загружена")
        except:
            self.collection = self.client.create_collection(
                name="documents",
                embedding_function=self.embedder
            )
            print("✓ Создана новая база данных")
            self.load_documents()
    
    def load_documents(self):
        """Загружает все PDF из папки в базу"""
        data_dir = Path(Config.DATA_DIR)
        if not data_dir.exists():
            print(f"✗ Папка {Config.DATA_DIR} не найдена. Создайте её и добавьте PDF файлы.")
            return
        
        pdf_files = list(data_dir.glob("*.pdf"))
        if not pdf_files:
            print(f"✗ В папке {Config.DATA_DIR} нет PDF файлов")
            return
        
        print(f"📚 Найдено {len(pdf_files)} PDF файлов")
        
        all_chunks = []
        all_metas = []
        all_ids = []
        chunk_id = 0
        
        for pdf_file in pdf_files:
            print(f"  Загружаю: {pdf_file.name}")
            text = extract_text_from_pdf(pdf_file)
            
            if not text:
                continue
                
            chunks = chunk_text(text)
            
            for chunk in chunks:
                all_chunks.append(chunk)
                all_metas.append({
                    "file": pdf_file.name,
                    "chunk": len(all_chunks)
                })
                all_ids.append(f"chunk_{chunk_id}")
                chunk_id += 1
        
        if all_chunks:
            self.collection.add(
                documents=all_chunks,
                metadatas=all_metas,
                ids=all_ids
            )
            print(f"✓ Загружено {len(all_chunks)} текстовых фрагментов")
    
    def search(self, query, n_results=5):
        """Ищет похожие тексты"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            
            # Форматируем результаты
            formatted = []
            for i in range(len(results['documents'][0])):
                doc = results['documents'][0][i]
                meta = results['metadatas'][0][i]
                score = 1 - results['distances'][0][i] if results['distances'] else 1.0
                
                formatted.append({
                    "text": doc[:500] + "..." if len(doc) > 500 else doc,
                    "file": meta.get("file", "unknown"),
                    "score": round(score, 3)
                })
            
            return formatted
        except Exception as e:
            print(f"Ошибка поиска: {e}")
            return []

# ============================================================================
# LLM API
# ============================================================================

def ask_ollama(prompt):
    """Запрос к локальной Ollama"""
    try:
        response = requests.post(
            Config.OLLAMA_URL,
            json={
                "model": Config.OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json().get("response", "Нет ответа")
        else:
            return f"Ошибка: {response.status_code}"
    except:
        return "Не удалось подключиться к Ollama. Убедитесь, что он запущен."

def ask_openrouter(prompt):
    """Запрос к OpenRouter"""
    try:
        headers = {
            "Authorization": f"Bearer {Config.OPENROUTER_KEY}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": Config.OPENROUTER_MODEL,
            "messages": [{"role": "user", "content": prompt}],
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
        else:
            return f"Ошибка API: {response.status_code}"
    except:
        return "Ошибка подключения к OpenRouter"

def ask_llm(prompt):
    """Выбор API для запроса"""
    if Config.API_CHOICE == "ollama":
        return ask_ollama(prompt)
    else:
        return ask_openrouter(prompt)

# ============================================================================
# RAG СИСТЕМА
# ============================================================================

class SimpleRAG:
    """Простая RAG система"""
    
    def __init__(self):
        print("🧠 Запуск RAG системы...")
        self.db = VectorDB()
    
    def system_prompt(self):
        """Системный промпт"""
        return """Ты — полезный ассистент, который отвечает на вопросы пользователя, 
используя предоставленные документы. Отвечай только на основе этих документов.
Если в документах нет информации — честно скажи об этом."""
    
    def ask(self, question):
        """Основной метод: задать вопрос системе"""
        print(f"\n❓ Вопрос: {question}")
        
        # Шаг 1: Поиск в базе знаний
        print("🔍 Ищу в документах...")
        results = self.db.search(question)
        
        if not results:
            return "В базе знаний нет информации по этому вопросу."
        
        # Шаг 2: Формируем контекст
        context = "ИНФОРМАЦИЯ ИЗ ДОКУМЕНТОВ:\n\n"
        for i, result in enumerate(results[:3]):  # Берём топ-3
            context += f"[Документ {i+1}, файл: {result['file']}]:\n"
            context += result['text'] + "\n\n"
        
        # Шаг 3: Создаём промпт
        prompt = f"""{self.system_prompt()}

{context}

ВОПРОС ПОЛЬЗОВАТЕЛЯ: {question}

ОТВЕТЬ на вопрос, используя ТОЛЬКО информацию из документов выше.
Если нужно — цитируй конкретные документы.
Если информации недостаточно — скажи об этом.

ОТВЕТ:"""
        
        print(f"Использованный промпт: {prompt}")
        # Шаг 4: Запрос к LLM
        print("🤖 Генерация ответа...")
        response = ask_llm(prompt)
        
        # Шаг 5: Форматируем финальный ответ
        final_response = f"{response}\n\n"
        final_response += "📚 Использованные источники:\n"
        for result in results[:3]:
            final_response += f"• {result['file']} (релевантность: {result['score']})\n"
        
        return final_response

# ============================================================================
# ЗАПУСК
# ============================================================================

def main():
    """Запуск системы"""
    print("=" * 50)
    print("ПРОСТАЯ RAG СИСТЕМА ДЛЯ ПОИСКА В PDF")
    print("=" * 50)
    
    # Создаём систему
    rag = SimpleRAG()
    
    # Примеры вопросов
    examples = [
        "Что такое искусственный интеллект?",
        "Какие методы машинного обучения существуют?",
        "Расскажи о нейронных сетях",
        "Что такое глубокое обучение?",
        "Выход"
    ]
    
    while True:
        print("\n" + "=" * 50)
        print("Выберите вариант:")
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example}")
        print("=" * 50)
        
        choice = input("Ваш выбор (1-5 или свой вопрос): ").strip()
        
        if choice == "5" or choice.lower() == "выход":
            print("👋 До свидания!")
            break
        
        if choice in ["1", "2", "3", "4"]:
            question = examples[int(choice) - 1]
        else:
            question = choice
        
        # Получаем ответ
        answer = rag.ask(question)
        
        # Выводим результат
        print("\n" + "=" * 50)
        print("💡 ОТВЕТ:")
        print("=" * 50)
        print(answer)

if __name__ == "__main__":
    # Создаём папку для данных, если её нет
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    
    # Запускаем
    main()
