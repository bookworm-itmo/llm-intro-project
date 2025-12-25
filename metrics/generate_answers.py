"""
Скрипт для генерации answers.csv с ответами через RAG пайплайн.
Создает два файла: answers_no_rerank.csv и answers_with_rerank.csv
"""
import json
import os
import sys
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Добавляем путь к корню проекта
sys.path.append(str(Path(__file__).parent.parent))

from services.rag_service.rag_engine import RAGEngine
from services.llm_service.claude_client import ClaudeClient

load_dotenv()


def generate_answers_csv(rag, llm, validation_data, output_path, use_reranker=False):
    """Генерирует answers.csv для заданного RAG движка."""
    print(f"\n{'='*60}")
    print(f"Генерация ответов {'С реранкером' if use_reranker else 'БЕЗ реранкера'}")
    print(f"{'='*60}")
    
    results = []
    
    for i, item in enumerate(tqdm(validation_data, desc="Обработка запросов"), 1):
        question = item["question"]
        expected_chapters = item.get("expected_chapters", [])
        
        # Получаем контекст через RAG
        retrieved = rag.search(question, top_k=5)
        
        if not retrieved:
            print(f"⚠ Нет результатов для вопроса: {question[:50]}...")
            continue
        
        # Формируем контекст из найденных чанков
        contexts = [r["text"] for r in retrieved]
        context_text = "\n\n".join(contexts)
        
        # Генерируем ответ через Claude
        try:
            answer = llm.generate_answer(
                query=question,
                context=context_text,
                sources=retrieved
            )
        except Exception as e:
            print(f"⚠ Ошибка генерации ответа для '{question[:50]}...': {e}")
            continue
        
        # Формируем ground_truth (ожидаемые главы как строка)
        gold_answers = ", ".join(str(ch) for ch in sorted(expected_chapters)) if expected_chapters else ""
        
        # Сохраняем в формате для CSV
        results.append({
            "query": question,
            "answer": answer,
            "context": json.dumps(contexts, ensure_ascii=False),  # JSON строка с массивом контекстов
            "gold_answers": gold_answers
        })
    
    # Сохраняем в CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n✓ Сохранено {len(results)} записей в {output_path}")
    
    return df


def main():
    print("Загрузка валидационной выборки...")
    validation_path = Path(__file__).parent.parent / "validation" / "validation_dataset.json"
    with open(validation_path, 'r', encoding='utf-8') as f:
        validation_data = json.load(f)
    print(f"Загружено {len(validation_data)} запросов")
    
    # Инициализация Claude для генерации ответов
    print("\nИнициализация Claude...")
    llm = ClaudeClient()
    print("✓ Claude инициализирован")
    
    # Инициализация RAG движков
    print("\nИнициализация RAG движков...")
    
    # RAG БЕЗ реранкера
    rag_no_rerank = RAGEngine(use_reranker=False)
    rag_no_rerank.load_chunks("data/chunks.parquet")
    rag_no_rerank.load_embeddings("data/embeddings.parquet")
    rag_no_rerank.load_index("data/faiss_index/index.faiss")
    print("✓ RAG без реранкера загружен")
    
    # RAG С реранкером
    rag_with_rerank = RAGEngine(use_reranker=True)
    rag_with_rerank.load_chunks("data/chunks.parquet")
    rag_with_rerank.load_embeddings("data/embeddings.parquet")
    rag_with_rerank.load_index("data/faiss_index/index.faiss")
    print("✓ RAG с реранкером загружен")
    
    # Генерируем answers.csv для обоих вариантов
    output_dir = Path(__file__).parent
    
    generate_answers_csv(
        rag_no_rerank,
        llm,
        validation_data,
        output_dir / "answers_no_rerank.csv",
        use_reranker=False
    )
    
    generate_answers_csv(
        rag_with_rerank,
        llm,
        validation_data,
        output_dir / "answers_with_rerank.csv",
        use_reranker=True
    )
    
    print("\n" + "="*60)
    print("ГОТОВО!")
    print("="*60)
    print(f"Созданы файлы:")
    print(f"  - {output_dir / 'answers_no_rerank.csv'}")
    print(f"  - {output_dir / 'answers_with_rerank.csv'}")


if __name__ == "__main__":
    main()

