import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from langchain_gigachat.embeddings import GigaChatEmbeddings
from tqdm import tqdm

load_dotenv()


class RAGEngine:
    def __init__(self):
        self.embeddings_client = GigaChatEmbeddings(
            credentials=os.getenv("GIGACHAT_AUTH_KEY"),
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=False
        )
        self.index = None
        self.chunks = []
        self.dimension = 1024  # GigaChat embeddings dimension

    def load_chunks(self, chunks_path: str):
        with open(chunks_path, 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)

    def _truncate_text(self, text: str, max_chars: int = 1000) -> str:
        """Обрезаем текст до максимального количества символов"""
        if len(text) <= max_chars:
            return text
        return text[:max_chars]

    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        # Обрезаем длинные тексты для соблюдения лимита токенов GigaChat
        truncated_texts = [self._truncate_text(t) for t in texts]

        # Обрабатываем батчами по 50 текстов
        all_embeddings = []
        batch_size = 50
        total_batches = (len(truncated_texts) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(truncated_texts), batch_size), total=total_batches, desc="Создание эмбеддингов"):
            batch = truncated_texts[i:i + batch_size]
            batch_embeddings = self.embeddings_client.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings, dtype='float32')

    def build_index(self):
        texts = [chunk['text'] for chunk in self.chunks]
        print(f"Создание эмбеддингов для {len(texts)} чанков...")
        embeddings = self.create_embeddings(texts)

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        print(f"Индекс создан. Размерность: {self.dimension}")

    def save_index(self, index_path: str):
        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)

    def load_index(self, index_path: str):
        self.index = faiss.read_index(index_path)

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_embedding = self.embeddings_client.embed_query(query)
        query_embedding = np.array([query_embedding], dtype='float32')
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx].copy()
                chunk['score'] = float(distance)
                results.append(chunk)

        return results

    def get_context_for_llm(self, query: str, top_k: int = 3) -> Tuple[str, List[Dict]]:
        results = self.search(query, top_k)

        context_parts = []
        for i, result in enumerate(results, 1):
            context_parts.append(
                f"[Глава {result['chapter']}]\n{result['text']}"
            )

        context = "\n\n".join(context_parts)
        return context, results
