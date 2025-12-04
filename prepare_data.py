from pathlib import Path
from services.data_service.download_book import check_book_exists, BOOK_PATH
from services.data_service.data_processor import BookProcessor
from services.rag_service.rag_engine import RAGEngine


def main():
    # Проверяем наличие книги
    print("Проверка наличия книги...")
    check_book_exists()

    print("Обработка книги...")
    processor = BookProcessor(str(BOOK_PATH))
    chunks = processor.process(chunk_size=300, overlap=30)
    processor.save_chunks("data/chunks.json")
    print(f"Создано {len(chunks)} чанков")

    print("Создание FAISS индекса...")
    rag = RAGEngine()
    rag.load_chunks("data/chunks.json")
    rag.build_index()
    rag.save_index("data/faiss_index/index.faiss")
    print("Индекс сохранен")


if __name__ == "__main__":
    main()
