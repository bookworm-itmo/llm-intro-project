import os
import numpy as np
import pandas as pd
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
        self.chunks_df = None
        self.embeddings_df = None
        self.dimension = 1024

    # ==================== Build Pipeline ====================

    def build_chunks(self, chunks: List[Dict], chunks_path: str) -> pd.DataFrame:
        """Сохраняет чанки в parquet."""
        df = pd.DataFrame(chunks)
        df['chunk_id'] = df['id'].astype('int64')
        df = df[['chunk_id', 'chapter', 'text']]

        Path(chunks_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(chunks_path, index=False)
        self.chunks_df = df
        return df

    def build_embeddings(self, embeddings_path: str) -> pd.DataFrame:
        """Создаёт эмбеддинги и сохраняет в parquet."""
        if self.chunks_df is None:
            raise ValueError("Сначала загрузите чанки")

        texts = self.chunks_df['text'].tolist()
        print(f"Создание эмбеддингов для {len(texts)} чанков...")

        # Создаём эмбеддинги батчами
        all_embeddings = []
        batch_size = 50
        total_batches = (len(texts) + batch_size - 1) // batch_size

        for i in tqdm(range(0, len(texts), batch_size), total=total_batches, desc="Создание эмбеддингов"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embeddings_client.embed_documents(batch)
            all_embeddings.extend(batch_embeddings)

        embeddings = np.array(all_embeddings, dtype='float32')

        # Нормализуем для косинусного сходства
        faiss.normalize_L2(embeddings)

        self.dimension = embeddings.shape[1]
        print(f"Размерность эмбеддингов: {self.dimension}")

        # Сохраняем в parquet
        emb_df = pd.DataFrame({
            'chunk_id': self.chunks_df['chunk_id'].astype('int64'),
            'embedding': list(embeddings)
        })

        Path(embeddings_path).parent.mkdir(parents=True, exist_ok=True)
        emb_df.to_parquet(embeddings_path, index=False)
        self.embeddings_df = emb_df
        return emb_df

    def build_faiss_index(self, index_path: str):
        """Строит FAISS IndexFlatIP и сохраняет."""
        if self.embeddings_df is None:
            raise ValueError("Сначала создайте эмбеддинги")

        embeddings = np.stack(self.embeddings_df['embedding'].to_numpy(), axis=0).astype('float32')

        # IndexFlatIP для косинусного сходства (эмбеддинги уже нормализованы)
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)

        Path(index_path).parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, index_path)
        print(f"Индекс сохранён: {len(embeddings)} векторов")

    # ==================== Load ====================

    def load_chunks(self, chunks_path: str):
        """Загружает чанки из parquet."""
        self.chunks_df = pd.read_parquet(chunks_path)

    def load_embeddings(self, embeddings_path: str):
        """Загружает эмбеддинги из parquet."""
        self.embeddings_df = pd.read_parquet(embeddings_path)

    def load_index(self, index_path: str):
        """Загружает FAISS индекс."""
        self.index = faiss.read_index(index_path)

    # ==================== Search ====================

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Поиск релевантных чанков по запросу."""
        # Получаем эмбеддинг запроса
        query_embedding = np.array(
            self.embeddings_client.embed_query(query),
            dtype='float32'
        ).reshape(1, -1)

        # Нормализуем для косинусного сходства
        faiss.normalize_L2(query_embedding)

        # Поиск (IndexFlatIP возвращает similarity, чем больше - тем лучше)
        similarities, indices = self.index.search(query_embedding, top_k)

        # Получаем chunk_id по индексу, затем тексты
        results = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if 0 <= idx < len(self.embeddings_df):
                chunk_id = self.embeddings_df.iloc[idx]['chunk_id']
                chunk_row = self.chunks_df[self.chunks_df['chunk_id'] == chunk_id].iloc[0]
                results.append({
                    'chunk_id': int(chunk_id),
                    'chapter': int(chunk_row['chapter']),
                    'text': chunk_row['text'],
                    'score': float(similarity)
                })

        return results

    def get_context_for_llm(self, query: str, top_k: int = 5) -> Tuple[str, List[Dict]]:
        """Получает контекст для LLM."""
        results = self.search(query, top_k)

        context_parts = []
        for result in results:
            context_parts.append(
                f"[Глава {result['chapter']}]\n{result['text']}"
            )

        context = "\n\n".join(context_parts)
        return context, results
