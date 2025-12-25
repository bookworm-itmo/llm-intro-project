"""
Оценка RAG системы с использованием RAGAS метрик.
Сравнивает качество с реранкером и без.
"""
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv
from tqdm import tqdm

# Подключаем модули проекта
sys.path.append(str(Path(__file__).parent.parent))
from services.rag_service.rag_engine import RAGEngine
from services.llm_service.claude_client import ClaudeClient

load_dotenv()

# Убираем deprecation warnings от RAGAS
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Импорт RAGAS
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]

# LLM через OpenRouter, эмбеддинги через GigaChat
from langchain_openai import ChatOpenAI
from langchain_gigachat.embeddings import GigaChatEmbeddings

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

RAGAS_LLM = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    temperature=0.0,
    max_tokens=1024,
)

RAGAS_EMBEDDINGS = GigaChatEmbeddings(
    credentials=os.getenv("GIGACHAT_AUTH_KEY"),
    scope="GIGACHAT_API_PERS",
    verify_ssl_certs=False
)

print("✓ RAGAS настроен (OpenRouter: GPT-4o-mini + GigaChat Embeddings)")


def load_rag_engine(use_reranker: bool) -> RAGEngine:
    """Загружает RAG движок."""
    rag = RAGEngine(use_reranker=use_reranker)
    rag.load_chunks("data/chunks.parquet")
    rag.load_embeddings("data/embeddings.parquet")
    rag.load_index("data/faiss_index/index.faiss")
    return rag


def generate_answers(rag: RAGEngine, llm: ClaudeClient, questions: list, top_k: int = 5) -> Dataset:
    """Генерирует ответы через RAG и формирует датасет для RAGAS."""
    rows = []

    for item in tqdm(questions, desc="Генерация ответов"):
        question = item["question"]
        gold_answer = item.get("gold_answer", "")

        # RAG поиск с retry при rate limit
        retrieved = None
        for attempt in range(5):
            try:
                retrieved = rag.search(question, top_k=top_k)
                break
            except Exception as e:
                if "429" in str(e) or "Too Many" in str(e):
                    wait_time = 10 * (attempt + 1)  # 10, 20, 30, 40, 50 сек
                    tqdm.write(f"⚠ Rate limit, жду {wait_time} сек (попытка {attempt+1}/5)...")
                    time.sleep(wait_time)
                else:
                    tqdm.write(f"⚠ Ошибка: {e}")
                    break

        if not retrieved:
            continue

        contexts = [r["text"] for r in retrieved]
        context_text = "\n\n".join(contexts)

        # Генерация ответа
        try:
            answer = llm.generate_answer(query=question, context=context_text, sources=retrieved)
        except Exception as e:
            tqdm.write(f"⚠ {e}")
            continue

        rows.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": gold_answer
        })

        # Задержка чтобы не упереться в rate limit GigaChat
        time.sleep(1.5)

    print(f"✓ Сгенерировано {len(rows)}/{len(questions)} ответов")
    return Dataset.from_list(rows)


def compute_metrics(dataset: Dataset) -> dict:
    """Вычисляет RAGAS метрики."""
    from ragas import evaluate

    try:
        result = evaluate(
            dataset=dataset,
            metrics=METRICS,
            llm=RAGAS_LLM,
            embeddings=RAGAS_EMBEDDINGS,
        )
        df = result.to_pandas()

        results = {}
        for metric in METRICS:
            name = metric.name
            if name in df.columns:
                scores = df[name].dropna().tolist()
                results[name] = {
                    "mean": float(np.mean(scores)) if scores else 0,
                    "median": float(np.median(scores)) if scores else 0,
                    "std": float(np.std(scores)) if scores else 0,
                    "min": float(np.min(scores)) if scores else 0,
                    "max": float(np.max(scores)) if scores else 0,
                    "scores": scores
                }
                print(f"  {name}: {results[name]['mean']:.3f}")
            else:
                print(f"  {name}: ⚠ не найден в результатах")
                results[name] = {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "scores": []}

        return results
    except Exception as e:
        print(f"⚠ Ошибка evaluate: {e}")
        return {m.name: {"mean": 0, "median": 0, "std": 0, "min": 0, "max": 0, "scores": []} for m in METRICS}


def run_evaluation(use_reranker: bool, questions: list, llm: ClaudeClient) -> dict:
    """Полный цикл оценки."""
    mode = "С реранкером" if use_reranker else "Без реранкера"
    print(f"\n{'='*60}\n{mode.upper()}\n{'='*60}")

    rag = load_rag_engine(use_reranker)
    dataset = generate_answers(rag, llm, questions)

    print("\nВычисление метрик:")
    return compute_metrics(dataset)


def print_comparison(results_no_rerank: dict, results_with_rerank: dict):
    """Выводит сравнительную таблицу."""
    print(f"\n{'='*60}\nСРАВНЕНИЕ\n{'='*60}")
    print(f"{'Метрика':<20} {'Без реранкера':>15} {'С реранкером':>15} {'Δ':>10}")
    print("-" * 60)

    for name in results_no_rerank:
        v1 = results_no_rerank[name]["mean"]
        v2 = results_with_rerank[name]["mean"]
        diff = v2 - v1
        print(f"{name:<20} {v1:>15.3f} {v2:>15.3f} {diff:>+10.3f}")


def save_results(results_no_rerank: dict, results_with_rerank: dict, output_path: str):
    """Сохраняет результаты в JSON."""
    output = {
        "without_reranker": results_no_rerank,
        "with_reranker": results_with_rerank
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Результаты сохранены: {output_path}")


def main():
    # Загрузка датасета
    dataset_path = Path(__file__).parent.parent / "metrics" / "dataset.csv"
    df = pd.read_csv(dataset_path)
    questions = [{"question": row["query"], "gold_answer": row["gold_answer"]} for _, row in df.iterrows()]
    print(f"Загружено {len(questions)} вопросов")

    llm = ClaudeClient()

    # Оценка
    results_no_rerank = run_evaluation(use_reranker=False, questions=questions, llm=llm)
    results_with_rerank = run_evaluation(use_reranker=True, questions=questions, llm=llm)

    # Результаты
    print_comparison(results_no_rerank, results_with_rerank)
    save_results(results_no_rerank, results_with_rerank, "validation/ragas_results.json")


if __name__ == "__main__":
    main()
