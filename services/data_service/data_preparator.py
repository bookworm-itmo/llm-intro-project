import json
from pathlib import Path
from typing import Dict

import faiss

from services.data_service.data_processor import BookProcessor
from services.rag_service.rag_engine import RAGEngine


class DataPreparator:
    """Класс для подготовки данных для RAG системы."""
    
    def __init__(
        self,
        data_dir: str = "data",
        chunk_size: int = 800,
        overlap: int = 100
    ):
        self.data_dir = Path(data_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.book_path = self.data_dir / "master_and_margarita.fb2"
        self.chunks_path = self.data_dir / "chunks.parquet"
        self.embeddings_path = self.data_dir / "embeddings.parquet"
        self.index_path = self.data_dir / "faiss_index" / "index.faiss"
        self.chunks_sample_json = self.data_dir / "chunks_sample.json"
        self.chunks_sample_txt = self.data_dir / "chunks_sample.txt"
    
    def verify_data_storage(self) -> bool:
        """Проверяет, что все данные сохранены в едином хранилище."""
        files_to_check = [
            self.book_path,
            self.chunks_path,
            self.embeddings_path,
            self.index_path
        ]
        
        return all(f.exists() for f in files_to_check)
    
    def prepare_all(self, verify: bool = True) -> bool:
        """Выполняет полный цикл подготовки данных."""
        if not self.book_path.exists():
            raise FileNotFoundError(f"Файл книги не найден: {self.book_path}")
        
        print("Подготовка данных...")
        
        processor = BookProcessor(str(self.book_path))
        text = processor.load_book()
        chapters = processor.split_into_chapters(text)
        chunks = processor.create_chunks(
            chunk_size=self.chunk_size,
            overlap=self.overlap
        )
        print(f"Создано {len(chunks)} чанков из {len(chapters)} глав")
        
        rag = RAGEngine()
        self.data_dir.mkdir(parents=True, exist_ok=True)
        rag.build_chunks(chunks, str(self.chunks_path))
        rag.build_embeddings(str(self.embeddings_path))
        rag.build_faiss_index(str(self.index_path))
        
        self._create_chunks_sample(chunks, chapters)
        
        if verify and not self.verify_data_storage():
            print("Ошибка: не все файлы созданы")
            return False
        
        print("Готово")
        return True
    
    def _create_chunks_sample(self, chunks: list, chapters: list):
        """Создает сэмплы чанков для отчета."""
        import pandas as pd
        
        df = pd.DataFrame(chunks)
        
        # JSON сэмпл
        chapters_to_sample = [1, 5, 10, 13, 15, 20, 23, 25, 30, 32]
        sample_data = {
            "metadata": {
                "total_chunks": len(chunks),
                "total_chapters": len(chapters),
                "sample_size": len(chapters_to_sample),
                "description": 'Примеры чанков из разных глав романа "Мастер и Маргарита"'
            },
            "chunks": []
        }
        
        for chapter in chapters_to_sample:
            chapter_chunks = df[df['chapter'] == chapter]
            if len(chapter_chunks) > 0:
                chunk = chapter_chunks.iloc[0]
                sample_data["chunks"].append({
                    "chunk_id": int(chunk['id']),
                    "chapter": int(chunk['chapter']),
                    "text": chunk['text'],
                    "text_length": len(chunk['text']),
                    "preview": chunk['text'][:150] + '...' if len(chunk['text']) > 150 else chunk['text']
                })
        
        with open(self.chunks_sample_json, 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        with open(self.chunks_sample_txt, 'w', encoding='utf-8') as f:
            f.write('СЭМПЛ ЧАНКОВ ИЗ РОМАНА "МАСТЕР И МАРГАРИТА"\n')
            f.write('=' * 80 + '\n\n')
            
            for chapter in chapters_to_sample:
                chapter_chunks = df[df['chapter'] == chapter]
                if len(chapter_chunks) > 0:
                    chunk = chapter_chunks.iloc[0]
                    f.write(f'ГЛАВА {int(chunk["chapter"])} (chunk_id={int(chunk["id"])})\n')
                    f.write('-' * 80 + '\n')
                    f.write(chunk['text'] + '\n')
                    f.write(f'\n[Длина: {len(chunk["text"])} символов]\n')
                    f.write('\n' + '=' * 80 + '\n\n')
    
    def check_vector_store_status(self) -> Dict[str, bool]:
        """Проверяет статус векторного хранилища."""
        status = {
            "chunks_exists": self.chunks_path.exists(),
            "embeddings_exists": self.embeddings_path.exists(),
            "index_exists": self.index_path.exists(),
            "all_ready": False
        }
        
        if (status["chunks_exists"] and
                status["embeddings_exists"] and
                status["index_exists"]):
            status["all_ready"] = True
            try:
                index = faiss.read_index(str(self.index_path))
                status["index_vectors"] = index.ntotal
                status["index_dimension"] = index.d
            except Exception as e:
                status["index_error"] = str(e)
        
        return status
