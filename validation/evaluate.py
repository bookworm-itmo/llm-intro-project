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


def evaluate_rag_single(rag: RAGEngine, validation_data: List[Dict], use_reranker: bool = False) -> Dict:
    """Оценка RAG системы с или без реранкера."""
    mode_name = "с реранкером" if use_reranker else "без реранкера"
    print(f"\n{'='*60}")
    print(f"Оценка RAG системы {mode_name.upper()}")
    print(f"{'='*60}\n")

    results = []
    total_precision = 0
    total_recall = 0
    total_f1 = 0

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
        print(f"   Найденные главы: {retrieved_chapters}")
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

    print(f"\n{'='*60}")
    print(f"СРЕДНИЕ МЕТРИКИ ({mode_name.upper()}):")
    print(f"Precision: {avg_metrics['avg_precision']:.3f}")
    print(f"Recall: {avg_metrics['avg_recall']:.3f}")
    print(f"F1: {avg_metrics['avg_f1']:.3f}")
    print(f"{'='*60}\n")

    return {
        "results": results,
        "avg_metrics": avg_metrics,
        "mode": mode_name
    }


def optimize_reranker_params(validation_data: List[Dict]) -> Dict:
    """Оптимизация гиперпараметров реранкера."""
    print("\n" + "="*60)
    print("ОПТИМИЗАЦИЯ ГИПЕРПАРАМЕТРОВ РЕРАНКЕРА")
    print("="*60)
    
    reranker_weights = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
    candidate_multipliers = [2.5, 3.0, 3.5, 4.0, 4.5]
    score_combinations = ["linear", "rrf", "geometric"]
    
    best_config = None
    best_f1 = 0.0
    best_metrics = None
    results_grid = []
    
    total_combinations = len(reranker_weights) * len(candidate_multipliers) * len(score_combinations)
    current = 0
    
    print(f"\nТестирование {total_combinations} комбинаций параметров...")
    print("Параметры: вес реранкера × множитель кандидатов × стратегия комбинации\n")
    
    for weight in reranker_weights:
        for multiplier in candidate_multipliers:
            for combo_strategy in score_combinations:
                current += 1
                print(f"[{current}/{total_combinations}] Вес: {weight:.2f}, Множитель: {multiplier:.1f}x, Стратегия: {combo_strategy}", end=" ... ")
                
                # Создаем новый экземпляр с параметрами
                rag = RAGEngine(
                    use_reranker=True, 
                    reranker_weight=weight,
                    rerank_candidate_multiplier=multiplier,
                    score_combination=combo_strategy
                )
            rag.load_chunks("data/chunks.parquet")
            rag.load_embeddings("data/embeddings.parquet")
            rag.load_index("data/faiss_index/index.faiss")
            
            # Быстрая оценка (без детального вывода)
            total_precision = 0
            total_recall = 0
            total_f1 = 0
            
            for item in validation_data:
                question = item["question"]
                expected_chapters = item["expected_chapters"]
                
                retrieved = rag.search(question, top_k=5)
                retrieved_chapters = [r["chapter"] for r in retrieved]
                metrics = calculate_metrics(retrieved_chapters, expected_chapters)
                
                total_precision += metrics['precision']
                total_recall += metrics['recall']
                total_f1 += metrics['f1']
            
            n = len(validation_data)
            avg_metrics = {
                "avg_precision": total_precision / n,
                "avg_recall": total_recall / n,
                "avg_f1": total_f1 / n
            }
            
            results_grid.append({
                "reranker_weight": weight,
                "candidate_multiplier": multiplier,
                "score_combination": combo_strategy,
                "metrics": avg_metrics
            })
            
            f1 = avg_metrics["avg_f1"]
            print(f"F1: {f1:.3f} (P: {avg_metrics['avg_precision']:.3f}, R: {avg_metrics['avg_recall']:.3f})")
            
            if f1 > best_f1:
                best_f1 = f1
                best_config = {
                    "reranker_weight": weight,
                    "candidate_multiplier": multiplier,
                    "score_combination": combo_strategy
                }
                best_metrics = avg_metrics
                print(f"  ✓ НОВЫЙ ЛУЧШИЙ РЕЗУЛЬТАТ!")
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ")
    print("="*60)
    print(f"\nЛучшая конфигурация:")
    print(f"  Вес реранкера: {best_config['reranker_weight']:.2f}")
    print(f"  Множитель кандидатов: {best_config['candidate_multiplier']:.1f}x")
    print(f"  Стратегия комбинации: {best_config['score_combination']}")
    print(f"\nМетрики:")
    print(f"  Precision: {best_metrics['avg_precision']:.3f}")
    print(f"  Recall: {best_metrics['avg_recall']:.3f}")
    print(f"  F1: {best_metrics['avg_f1']:.3f}")
    print("="*60)
    
    return {
        "best_config": best_config,
        "best_metrics": best_metrics,
        "all_results": results_grid
    }


def evaluate_rag():
    """Оценка RAG системы с сравнением метрик с/без реранкера."""
    print("Загрузка валидационной выборки...")
    with open("validation/validation_dataset.json", 'r', encoding='utf-8') as f:
        validation_data = json.load(f)

    print("\nЗагрузка RAG системы (без реранкера)...")
    rag_no_rerank = RAGEngine(use_reranker=False)
    rag_no_rerank.load_chunks("data/chunks.parquet")
    rag_no_rerank.load_embeddings("data/embeddings.parquet")
    rag_no_rerank.load_index("data/faiss_index/index.faiss")

    results_no_rerank = evaluate_rag_single(rag_no_rerank, validation_data, use_reranker=False)

    optimization_results = optimize_reranker_params(validation_data)
    best_config = optimization_results["best_config"]
    
    print("\n" + "="*60)
    print("ОЦЕНКА С ОПТИМИЗИРОВАННЫМ РЕРАНКЕРОМ")
    print("="*60)
    rag_with_rerank = RAGEngine(
        use_reranker=True, 
        reranker_weight=best_config["reranker_weight"],
        rerank_candidate_multiplier=best_config["candidate_multiplier"],
        score_combination=best_config["score_combination"]
    )
    rag_with_rerank.load_chunks("data/chunks.parquet")
    rag_with_rerank.load_embeddings("data/embeddings.parquet")
    rag_with_rerank.load_index("data/faiss_index/index.faiss")

    results_with_rerank = evaluate_rag_single(rag_with_rerank, validation_data, use_reranker=True)

    print("\n" + "="*60)
    print("СРАВНЕНИЕ МЕТРИК")
    print("="*60)
    
    metrics_no_rerank = results_no_rerank["avg_metrics"]
    metrics_with_rerank = results_with_rerank["avg_metrics"]
    
    print(f"\n{'Метрика':<20} {'Без реранкера':<20} {'С реранкером':<20} {'Изменение':<20}")
    print("-" * 80)
    
    for metric in ["avg_precision", "avg_recall", "avg_f1"]:
        name = metric.replace("avg_", "").capitalize()
        no_rerank_val = metrics_no_rerank[metric]
        with_rerank_val = metrics_with_rerank[metric]
        diff = with_rerank_val - no_rerank_val
        diff_pct = (diff / no_rerank_val * 100) if no_rerank_val > 0 else 0
        diff_str = f"{diff:+.3f} ({diff_pct:+.1f}%)"
        
        print(f"{name:<20} {no_rerank_val:<20.3f} {with_rerank_val:<20.3f} {diff_str:<20}")
    
    print("="*60)
    
    f1_no_rerank = metrics_no_rerank["avg_f1"]
    f1_with_rerank = metrics_with_rerank["avg_f1"]
    
    if f1_with_rerank > f1_no_rerank:
        print(f"\n✓ Реранкер УЛУЧШАЕТ метрики (F1: {f1_no_rerank:.3f} → {f1_with_rerank:.3f})")
    elif f1_with_rerank < f1_no_rerank:
        print(f"\n✗ Реранкер УХУДШАЕТ метрики (F1: {f1_no_rerank:.3f} → {f1_with_rerank:.3f})")
    else:
        print(f"\n= Реранкер не влияет на метрики (F1: {f1_no_rerank:.3f})")
    
    comparison_results = {
        "without_reranker": results_no_rerank,
        "with_reranker": results_with_rerank,
        "optimization": optimization_results,
        "comparison": {
            "precision_improvement": metrics_with_rerank["avg_precision"] - metrics_no_rerank["avg_precision"],
            "recall_improvement": metrics_with_rerank["avg_recall"] - metrics_no_rerank["avg_recall"],
            "f1_improvement": metrics_with_rerank["avg_f1"] - metrics_no_rerank["avg_f1"],
            "best_mode": "with_reranker" if f1_with_rerank > f1_no_rerank else "without_reranker",
            "best_config": best_config
        }
    }
    
    with open("validation/results.json", 'w', encoding='utf-8') as f:
        json.dump(comparison_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты сохранены в validation/results.json")
    
    return comparison_results


if __name__ == "__main__":
    evaluate_rag()
