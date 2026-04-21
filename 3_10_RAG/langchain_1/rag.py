#from markitdown import MarkItDown
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

import chromadb

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=100,
    separators=["\n\n", "\n", ". ", ", ", " "]
)

#md = MarkItDown()
import fitz

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

all_chunks = []
pdf_dir = "pdf/"

client = chromadb.PersistentClient(path="./vector_db")
collection = client.get_or_create_collection(name="patents")

model = SentenceTransformer('intfloat/multilingual-e5-small') #'multi-qa-mpnet-base-dot-v1'

if (collection.count() == 0):

    for filename in os.listdir(pdf_dir):
        if filename.endswith(".pdf"):
            file_path = os.path.join(pdf_dir, filename)
            print(f"Processing {filename}...")
            
            # Конвертация и разбивка
            #result = md.convert(file_path)
            result = extract_text_from_pdf(file_path)
            chunks = text_splitter.split_text(result)
            chunks_with_prefix = [f"passage: {chunk}" for chunk in chunks]
            all_chunks.extend(chunks_with_prefix)

    embeddings = model.encode(chunks, normalize_embeddings = True)

    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"chunk_{i}" for i in range(len(chunks))]
    )

query = "query: Как работает сверточная нейронная сеть?"
query_embedding = model.encode([query], normalize_embeddings = True)

results = collection.query(
    query_embeddings=query_embedding.tolist(),
    n_results=7,
    include=["documents"]
)

llm = OllamaLLM(model="hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF")
prompt = PromptTemplate(
    template="Ответь на вопрос на основе контекста:\nКонтекст: {context}\nВопрос: {question}\nОтвет:",
    input_variables=["context", "question"]
)

context = "\n".join(results["documents"][0])
print(f"RAG-контекст: {context}")

answer = llm.invoke(prompt.format(context=context, question=query))
print(answer)