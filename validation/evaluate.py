import json
import sys
from pathlib import Path
from typing import List, Dict

sys.path.append(str(Path(__file__).parent.parent))

from services.rag_service.rag_engine import RAGEngine
from services.llm_service.claude_client import ClaudeClient


def calculate_metrics(
    retrieved_chapters: List[int],
    expected_chapters: List[int]
) -> Dict[str, float]:
    retrieved_set = set(retrieved_chapters)
    expected_set = set(expected_chapters)

    if len(retrieved_set) == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    true_positives = len(retrieved_set & expected_set)
    precision = true_positives / len(retrieved_set) if retrieved_set else 0
    recall = true_positives / len(expected_set) if expected_set else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def evaluate_rag():
    print("Загрузка RAG системы...")
    rag = RAGEngine()
    rag.load_chunks("data/chunks.parquet")
    rag.load_embeddings("data/embeddings.parquet")
    rag.load_index("data/faiss_index/index.faiss")

    print("Загрузка валидационной выборки...")
    with open("validation/validation_dataset.json", 'r', encoding='utf-8') as f:
        validation_data = json.load(f)

    results = []
    total_precision = 0
    total_recall = 0
    total_f1 = 0

    print("\nОценка RAG системы:\n")

    for i, item in enumerate(validation_data, 1):
        question = item["question"]
        expected_chapters = item["expected_chapters"]

        retrieved = rag.search(question, top_k=5)
        retrieved_chapters = [r["chapter"] for r in retrieved]

        metrics = calculate_metrics(retrieved_chapters, expected_chapters)

        results.append({
            "question": question,
            "expected_chapters": expected_chapters,
            "retrieved_chapters": retrieved_chapters,
            "metrics": metrics
        })

        print(f"{i}. {question}")
        print(f"   Ожидаемые главы: {expected_chapters}")
        print(f"   Найденные главы: {retrieved_chapters[:5]}")
        print(f"   Precision: {metrics['precision']:.3f}")
        print(f"   Recall: {metrics['recall']:.3f}")
        print(f"   F1: {metrics['f1']:.3f}\n")

        total_precision += metrics['precision']
        total_recall += metrics['recall']
        total_f1 += metrics['f1']

    n = len(validation_data)
    avg_metrics = {
        "avg_precision": total_precision / n,
        "avg_recall": total_recall / n,
        "avg_f1": total_f1 / n
    }

    print("\n" + "="*50)
    print("СРЕДНИЕ МЕТРИКИ:")
    print(f"Precision: {avg_metrics['avg_precision']:.3f}")
    print(f"Recall: {avg_metrics['avg_recall']:.3f}")
    print(f"F1: {avg_metrics['avg_f1']:.3f}")
    print("="*50)

    with open("validation/results.json", 'w', encoding='utf-8') as f:
        json.dump({
            "results": results,
            "avg_metrics": avg_metrics
        }, f, ensure_ascii=False, indent=2)

    return avg_metrics


if __name__ == "__main__":
    evaluate_rag()
