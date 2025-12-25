import os
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
from langchain_gigachat.embeddings import GigaChatEmbeddings
from tqdm import tqdm

load_dotenv()


class RAGEngine:
    def __init__(
        self, 
        use_reranker: bool = False, 
        reranker_model: Optional[str] = None,
        reranker_weight: float = 0.7,
        rerank_candidate_multiplier: float = 3.0,
        score_combination: str = "linear"
    ):
        """
        Инициализация RAG движка.
        
        Args:
            use_reranker: Использовать ли реранкер для улучшения результатов поиска
            reranker_model: Название модели реранкера (по умолчанию 'BAAI/bge-reranker-base' - мультиязычная)
            reranker_weight: Вес реранкера в гибридном скоре (0.0-1.0). 0.7 означает 70% реранкер, 30% эмбеддинги
            rerank_candidate_multiplier: Множитель для количества кандидатов при реранкинге (по умолчанию 3.0 = top_k * 3)
            score_combination: Стратегия комбинации скоров: "linear", "rrf" (Reciprocal Rank Fusion), "geometric"
        """
        self.embeddings_client = GigaChatEmbeddings(
            credentials=os.getenv("GIGACHAT_AUTH_KEY"),
            scope="GIGACHAT_API_PERS",
            verify_ssl_certs=False
        )
        self.index = None
        self.chunks_df = None
        self.embeddings_df = None
        self.dimension = 1024
        self.use_reranker = use_reranker
        self.reranker = None
        self.reranker_weight = reranker_weight
        self.rerank_candidate_multiplier = rerank_candidate_multiplier
        self.score_combination = score_combination
        
        if use_reranker:
            try:
                from sentence_transformers import CrossEncoder
                if reranker_model is None:
                    models_to_try = [
                        'BAAI/bge-reranker-base',
                        'jinaai/jina-reranker-v1-base-en',
                        'cross-encoder/ms-marco-MiniLM-L-6-v2',
                    ]
                else:
                    models_to_try = [reranker_model]
                
                self.reranker = None
                for model_name in models_to_try:
                    try:
                        print(f"Загрузка реранкера: {model_name}...")
                        self.reranker = CrossEncoder(model_name)
                        print(f"✓ Реранкер загружен успешно: {model_name} (вес: {reranker_weight:.1%})")
                        break
                    except Exception as e:
                        print(f"  Не удалось загрузить {model_name}: {e}")
                        if model_name == models_to_try[-1]:
                            raise
                        continue
                
                if self.reranker is None:
                    raise ValueError("Не удалось загрузить ни одну модель реранкера")
                    
            except ImportError:
                print("Предупреждение: sentence-transformers не установлен. Реранкер отключен.")
                self.use_reranker = False
            except Exception as e:
                print(f"Ошибка при загрузке реранкера: {e}. Реранкер отключен.")
                self.use_reranker = False


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

    def search(self, query: str, top_k: int = 5, rerank_top_k: Optional[int] = None) -> List[Dict]:
        """
        Поиск релевантных чанков по запросу.
        
        Args:
            query: Текст запроса
            top_k: Количество результатов для возврата
            rerank_top_k: Количество кандидатов для реранкинга (если None, используется top_k * 2)
        """
        # Получаем эмбеддинг запроса
        query_embedding = np.array(
            self.embeddings_client.embed_query(query),
            dtype='float32'
        ).reshape(1, -1)

        # Нормализуем для косинусного сходства
        faiss.normalize_L2(query_embedding)

        if self.use_reranker and rerank_top_k is None:
            search_k = max(int(top_k * self.rerank_candidate_multiplier), 15)
        elif rerank_top_k is not None:
            search_k = rerank_top_k
        else:
            search_k = top_k

        # Поиск (IndexFlatIP возвращает similarity, чем больше - тем лучше)
        similarities, indices = self.index.search(query_embedding, search_k)

        # Получаем chunk_id по индексу, затем тексты
        candidates = []
        for idx, similarity in zip(indices[0], similarities[0]):
            if 0 <= idx < len(self.embeddings_df):
                chunk_id = self.embeddings_df.iloc[idx]['chunk_id']
                chunk_row = self.chunks_df[self.chunks_df['chunk_id'] == chunk_id].iloc[0]
                candidates.append({
                    'chunk_id': int(chunk_id),
                    'chapter': int(chunk_row['chapter']),
                    'text': chunk_row['text'],
                    'score': float(similarity)
                })

        if self.use_reranker and self.reranker is not None and len(candidates) > top_k:
            results = self._rerank(query, candidates, top_k)
        else:
            results = candidates[:top_k]

        return results

    def _rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        Применяет реранкер к кандидатам с гибридной стратегией комбинации скоров.
        
        Использует взвешенную комбинацию скоров эмбеддингов и реранкера для лучших результатов.
        """
        if not candidates:
            return []

        pairs = [[query, candidate['text']] for candidate in candidates]

        rerank_scores = self.reranker.predict(pairs, show_progress_bar=False)

        embedding_scores = np.array([c['score'] for c in candidates])

        if embedding_scores.min() < 0:
            embedding_scores = (embedding_scores - embedding_scores.min()) / (embedding_scores.max() - embedding_scores.min() + 1e-8)
        else:
            if embedding_scores.max() > 0:
                embedding_scores = embedding_scores / embedding_scores.max()

        rerank_scores_np = np.array(rerank_scores)
        if rerank_scores_np.min() < 0:
            rerank_scores_np = (rerank_scores_np - rerank_scores_np.min()) / (rerank_scores_np.max() - rerank_scores_np.min() + 1e-8)
        else:
            if rerank_scores_np.max() > 0:
                rerank_scores_np = rerank_scores_np / rerank_scores_np.max()

        if self.score_combination == "rrf":
            embedding_ranks = np.argsort(np.argsort(-embedding_scores)) + 1
            rerank_ranks = np.argsort(np.argsort(-rerank_scores_np)) + 1
            
            k = 60
            embedding_weight = 1.0 - self.reranker_weight
            hybrid_scores = (
                embedding_weight / (k + embedding_ranks) +
                self.reranker_weight / (k + rerank_ranks)
            )
        elif self.score_combination == "geometric":
            embedding_weight = 1.0 - self.reranker_weight
            log_emb = np.log(embedding_scores + 1e-8)
            log_rerank = np.log(rerank_scores_np + 1e-8)
            hybrid_scores = np.exp(
                embedding_weight * log_emb + 
                self.reranker_weight * log_rerank
            )
        else:
            embedding_weight = 1.0 - self.reranker_weight
            hybrid_scores = (
                embedding_weight * embedding_scores + 
                self.reranker_weight * rerank_scores_np
            )

        reranked = []
        for candidate, emb_score, rerank_score, hybrid_score in zip(
            candidates, embedding_scores, rerank_scores_np, hybrid_scores
        ):
            reranked.append({
                **candidate,
                'original_score': float(candidate['score']),
                'normalized_embedding_score': float(emb_score),
                'rerank_score': float(rerank_score),
                'hybrid_score': float(hybrid_score),
                'score': float(hybrid_score)
            })

        reranked.sort(key=lambda x: x['hybrid_score'], reverse=True)

        return reranked[:top_k]

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
