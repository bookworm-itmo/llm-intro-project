from pathlib import Path
from services.data_service.download_book import check_book_exists, BOOK_PATH
from services.data_service.data_processor import BookProcessor
from services.rag_service.rag_engine import RAGEngine

# Пути к данным
DATA_DIR = Path("data")
CHUNKS_PATH = DATA_DIR / "chunks.parquet"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.parquet"
INDEX_PATH = DATA_DIR / "faiss_index" / "index.faiss"


def main():
    # 1. Проверяем наличие книги
    print("=" * 50)
    print("Шаг 1: Проверка книги")
    print("=" * 50)
    check_book_exists()

    # 2. Создаём чанки
    print("\n" + "=" * 50)
    print("Шаг 2: Создание чанков")
    print("=" * 50)
    processor = BookProcessor(str(BOOK_PATH))
    chunks = processor.process(chunk_size=800, overlap=100)
    print(f"Создано {len(chunks)} чанков")

    # 3. RAG Pipeline: chunks → embeddings → index
    print("\n" + "=" * 50)
    print("Шаг 3: RAG Pipeline")
    print("=" * 50)

    rag = RAGEngine()

    # 3.1 Сохраняем чанки в parquet
    print("\n[3.1] Сохранение чанков в parquet...")
    rag.build_chunks(chunks, str(CHUNKS_PATH))
    print(f"Чанки сохранены: {CHUNKS_PATH}")

    # 3.2 Создаём эмбеддинги
    print("\n[3.2] Создание эмбеддингов...")
    rag.build_embeddings(str(EMBEDDINGS_PATH))
    print(f"Эмбеддинги сохранены: {EMBEDDINGS_PATH}")

    # 3.3 Строим FAISS индекс
    print("\n[3.3] Построение FAISS индекса...")
    rag.build_faiss_index(str(INDEX_PATH))
    print(f"Индекс сохранён: {INDEX_PATH}")

    print("\n" + "=" * 50)
    print("Готово!")
    print("=" * 50)


if __name__ == "__main__":
    main()
