import os
import re
import requests
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
import fitz  # PyMuPDF
from decouple import config
from rank_bm25 import BM25Okapi  # NEW: BM25 reranking

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

class Config:
    # Выбор API: "ollama" или "openrouter"
    API_CHOICE = "openrouter"  # поменяйте при необходимости

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
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"

    # NEW: Hybrid BM25 reranking config
    BM25_WEIGHT = 0.5  # 0.0 = только Embedding, 1.0 = только BM25
    HYBRID_ENABLED = True  # вкл/выкл гибридный режим

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
        self.bm25 = None  # NEW: BM25 index (для reranking)
        self.corpus_texts = []  # NEW: тексты чанков в той же последовательности
        self.bm25_tokenized = []  # NEW: токенизированные чанки для BM25
        self.corpus_bm25_scores_cache = None  # NEW: кэш скоров BM25 для текущего запроса (не обязателен, можно перерасчитывать)

        try:
            self.collection = self.client.get_collection("documents", embedding_function=self.embedder)
            print("✓ База данных загружена")
        except:
            self.collection = self.client.create_collection(
                name="documents",
                embedding_function=self.embedder,
                 configuration={
                    "hnsw": {
                        "space": "cosine"
                    }
                 }
            )
            print("✓ Создана новая база данных")
            self.load_documents()
    
    def _tokenize_text(self, text):
        # Простая токенизация: lowercased by words
        return text.lower().split()
    
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

            # NEW: Build BM25 index for reranking on top of embedding search
            # Tokenize corpus for BM25
            self.corpus_texts = all_chunks
            self.bm25_tokenized = [c.split() for c in all_chunks]
            if self.bm25_tokenized:
                self.bm25 = BM25Okapi(self.bm25_tokenized)
                print("✓ BM25 индекс создан для гибридного поиска")
    
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
                score_embed = 1 - results['distances'][0][i] if results['distances'] else 1.0
                
                formatted.append({
                    "text": doc, #doc[:500] + "..." if len(doc) > 500 else doc,
                    "file": meta.get("file", "unknown"),
                    "chunk": meta.get("chunk", None),
                    "emb_score": score_embed
                })
            
            # NEW: BM25 reranking (hybrid)
            if Config.HYBRID_ENABLED and self.bm25 is not None and self.corpus_texts:
                # Prepare BM25 scores for the query against entire corpus
                q_tokens = self._tokenize_text(query)
                bm25_scores = self.bm25.get_scores(q_tokens)  # aligned to corpus_texts
                max_bm25 = max(bm25_scores) if len(bm25_scores) > 0 else 1.0

                # Compute hybrid score for each result, then sort by it
                hybrid_results = []
                for idx, item in enumerate(formatted):
                    chunk_idx = (item["chunk"] - 1) if isinstance(item.get("chunk"), int) else None
                    if chunk_idx is not None and 0 <= chunk_idx < len(bm25_scores):
                        bm25_score_norm = bm25_scores[chunk_idx] / (max_bm25 + 1e-9)
                    else:
                        bm25_score_norm = 0.0

                    emb_score = item.get("emb_score", 0.0)
                    final_score = (Config.BM25_WEIGHT * bm25_score_norm) + ((1.0 - Config.BM25_WEIGHT) * emb_score)

                    hybrid_results.append({
                        "text": item["text"],
                        "file": item["file"],
                        "chunk": item.get("chunk"),
                        "emb_score": emb_score,
                        "bm25_score_norm": bm25_score_norm,
                        "score": final_score
                    })

                # Sort by hybrid score (desc)
                hybrid_results.sort(key=lambda x: x["score"], reverse=True)

                # Return in the expected format
                return [
                    {"text": r["text"], "file": r["file"], "score": round(r["score"], 3)}
                    for r in hybrid_results
                ]

            # If hybrid not enabled or BM25 not available, return embedding-based results
            for i in range(len(formatted)):
                formatted[i]["score"] = round(formatted[i].get("emb_score", 0.0), 3)

            # Fallback: sort by embedding score (already in order, but ensure sorting)
            formatted.sort(key=lambda x: x["score"], reverse=True)

            return [
                {"text": r["text"], "file": r["file"], "score": r["score"]}
                for r in formatted
            ]
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
        for i, result in enumerate(results):
            context += f"[Документ {i+1}, файл: {result['file']}]:\n"
            context += f"{result['text']}\n\n"
        
        prompt = self.system_prompt() + "\n\n" + context + "\nПожалуйста, ответь на вопрос: " + question

        print(f"Использованный промпт: {prompt}")
        
        # Шаг 3: Ответ от LLM
        answer = ask_llm(prompt)
        return answer

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

#Как использовать
#- Установите зависимость:
#  - pip install rank_bm25
#- По умолчанию включен гибридный режим (HYBRID_ENABLED = True) и вес BM25_BM25_WEIGHT = 0.5. Чтобы полностью полагаться на векторное поиск, установите BM25_WEIGHT = 0.0 или HYBRID_ENABLED = False.
#- Прогоните ваш код как обычно. При загрузке PDF будет построен BM25 индекс для reranking.

#Объяснение
#- Векторная часть остается как прежде: сначала выполняется поиск по эмбеддингам в ChromaDB.
#- Новая часть: на основе BM25 рассчитывается релевантность фрагментов к запросу. BM25 scores нормализуются по текущему набору результатов (через max_score) и комбинируются с Embedding score в гибридный score.
#- Результаты сортируются по гибридному score, возвращая наиболее релевантные фрагменты документов.