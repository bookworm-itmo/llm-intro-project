import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
from datasets import Dataset
import numpy as np

sys.path.append(str(Path(__file__).parent.parent))

from services.rag_service.rag_engine import RAGEngine
from services.llm_service.claude_client import ClaudeClient

# Проверка доступности RAGAS
RAGAS_AVAILABLE = False
METRICS_TO_EVALUATE = []

try:
    # Пытаемся импортировать метрики из нового API (ragas >= 0.1.0)
    try:
        from ragas.metrics.collections import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            context_utilization
        )
        METRICS_TO_EVALUATE = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            context_utilization
        ]
        RAGAS_AVAILABLE = True
        print("✓ RAGAS метрики загружены из ragas.metrics.collections")
    except ImportError:
        # Fallback для старых версий
        try:
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            )
            METRICS_TO_EVALUATE = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall
            ]
            # Пытаемся импортировать context_utilization, если доступен
            try:
                from ragas.metrics import context_utilization
                METRICS_TO_EVALUATE.append(context_utilization)
            except ImportError:
                print("⚠ context_utilization недоступен в этой версии RAGAS")
            RAGAS_AVAILABLE = True
            print("✓ RAGAS метрики загружены из ragas.metrics")
        except ImportError as e:
            print(f"⚠ Не удалось импортировать RAGAS метрики: {e}")
            RAGAS_AVAILABLE = False
except Exception as e:
    print(f"⚠ Ошибка при импорте RAGAS: {e}")
    RAGAS_AVAILABLE = False

# Настройка LLM и embeddings для RAGAS
try:
    from langchain_gigachat.llms import GigaChat
    from langchain_gigachat.embeddings import GigaChatEmbeddings
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Инициализация для RAGAS
    ragas_llm = GigaChat(
        credentials=os.getenv("GIGACHAT_AUTH_KEY"),
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,
        model="GigaChat-Pro",
        temperature=0.1
    )
    
    ragas_embeddings = GigaChatEmbeddings(
        credentials=os.getenv("GIGACHAT_AUTH_KEY"),
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False
    )
    
    RAGAS_CONFIGURED = True
except Exception as e:
    print(f"⚠ Не удалось настроить GigaChat для RAGAS: {e}")
    RAGAS_CONFIGURED = False


def prepare_dataset_for_ragas(
    rag: RAGEngine,
    llm_client: ClaudeClient,
    validation_data: List[Dict],
    top_k: int = 5
) -> Dataset:
    """Подготовка датасета для оценки RAGAS."""
    print(f"\nПодготовка датасета для RAGAS ({len(validation_data)} запросов)...")
    
    dataset_rows = []
    
    for i, item in enumerate(validation_data, 1):
        question = item["question"]
        expected_chapters = item.get("expected_chapters", [])
        
        print(f"[{i}/{len(validation_data)}] Обработка: {question[:50]}...", end=" ")
        
        # Получаем контекст через RAG
        retrieved = rag.search(question, top_k=top_k)
        
        if not retrieved:
            print("⚠ Нет результатов поиска")
            continue
        
        # Формируем контекст из найденных чанков
        contexts = [r["chunk"] for r in retrieved]
        context_text = "\n\n".join(contexts)
        
        # Генерируем ответ через LLM
        try:
            answer = llm_client.generate_answer(
                query=question,
                context=context_text,
                sources=retrieved
            )
        except Exception as e:
            print(f"⚠ Ошибка генерации ответа: {e}")
            continue
        
        # Формируем ground_truth (ожидаемые главы как строка)
        ground_truth = ", ".join(str(ch) for ch in sorted(expected_chapters)) if expected_chapters else ""
        
        dataset_rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth
        })
        
        print("✓")
    
    if not dataset_rows:
        raise ValueError("Не удалось подготовить данные для RAGAS")
    
    print(f"✓ Подготовлено {len(dataset_rows)} записей")
    
    return Dataset.from_list(dataset_rows)


def evaluate_with_ragas(
    rag: RAGEngine,
    llm_client: ClaudeClient,
    validation_data: List[Dict],
    use_reranker: bool = False
) -> Dict:
    """Оценка RAG системы с использованием RAGAS метрик."""
    if not RAGAS_AVAILABLE:
        raise RuntimeError("RAGAS не установлен или не может быть импортирован")
    
    if not RAGAS_CONFIGURED:
        raise RuntimeError("GigaChat не настроен для RAGAS")
    
    mode_name = "с реранкером" if use_reranker else "без реранкера"
    print(f"\n{'='*60}")
    print(f"ОЦЕНКА RAGAS {mode_name.upper()}")
    print(f"{'='*60}")
    
    # Подготавливаем датасет
    dataset = prepare_dataset_for_ragas(rag, llm_client, validation_data)
    
    # Вычисляем метрики
    print(f"\nВычисление метрик RAGAS...")
    print(f"Используемые метрики: {[m.name for m in METRICS_TO_EVALUATE if m is not None]}")
    
    results = {}
    
    for metric in METRICS_TO_EVALUATE:
        if metric is None:
            continue
        
        try:
            print(f"  Вычисление {metric.name}...", end=" ")
            
            # Настраиваем метрику с LLM и embeddings
            metric.llm = ragas_llm
            metric.embeddings = ragas_embeddings
            
            # Вычисляем метрику
            scores = metric.score(dataset)
            results[metric.name] = scores
            
            print(f"✓ (среднее: {np.mean(scores):.3f})")
        except Exception as e:
            print(f"⚠ Ошибка: {e}")
            results[metric.name] = []
    
    # Вычисляем статистику
    stats = {
        'mean': {},
        'median': {},
        'std': {},
        'min': {},
        'max': {}
    }
    
    for metric_name, scores in results.items():
        if scores and len(scores) > 0:
            stats['mean'][metric_name] = float(np.mean(scores))
            stats['median'][metric_name] = float(np.median(scores))
            stats['std'][metric_name] = float(np.std(scores))
            stats['min'][metric_name] = float(np.min(scores))
            stats['max'][metric_name] = float(np.max(scores))
        else:
            stats['mean'][metric_name] = 0.0
            stats['median'][metric_name] = 0.0
            stats['std'][metric_name] = 0.0
            stats['min'][metric_name] = 0.0
            stats['max'][metric_name] = 0.0
    
    return {
        'raw_scores': results,
        **stats
    }


def main():
    if not RAGAS_AVAILABLE:
        print("\n" + "="*60)
        print("ОШИБКА: RAGAS не установлен!")
        print("Установите зависимости:")
        print("  pip install ragas langchain-community datasets")
        print("="*60)
        return
    
    if not RAGAS_CONFIGURED:
        print("\n" + "="*60)
        print("ОШИБКА: GigaChat не настроен для RAGAS!")
        print("Проверьте переменную окружения GIGACHAT_AUTH_KEY")
        print("="*60)
        return
    
    print("Загрузка валидационной выборки...")
    with open("validation/validation_dataset.json", 'r', encoding='utf-8') as f:
        validation_data = json.load(f)
    print(f"Загружено {len(validation_data)} запросов")
    
    llm_client = ClaudeClient()  # Инициализируем ClaudeClient один раз
    
    # Оценка БЕЗ реранкера
    print("\n" + "="*60)
    print("ОЦЕНКА БЕЗ РЕРАНКЕРА")
    print("="*60)
    
    rag_no_rerank = RAGEngine(use_reranker=False)
    rag_no_rerank.load_chunks("data/chunks.parquet")
    rag_no_rerank.load_embeddings("data/embeddings.parquet")
    rag_no_rerank.load_index("data/faiss_index/index.faiss")
    
    results_no_rerank = evaluate_with_ragas(
        rag_no_rerank, llm_client, validation_data, use_reranker=False
    )
    
    # Оценка С реранкером
    print("\n" + "="*60)
    print("ОЦЕНКА С РЕРАНКЕРОМ")
    print("="*60)
    
    rag_with_rerank = RAGEngine(use_reranker=True)
    rag_with_rerank.load_chunks("data/chunks.parquet")
    rag_with_rerank.load_embeddings("data/embeddings.parquet")
    rag_with_rerank.load_index("data/faiss_index/index.faiss")
    
    results_with_rerank = evaluate_with_ragas(
        rag_with_rerank, llm_client, validation_data, use_reranker=True
    )
    
    # Выводим результаты
    print("\n" + "="*60)
    print("СРАВНЕНИЕ МЕТРИК RAGAS")
    print("="*60)
    
    metrics_names = [m.name for m in METRICS_TO_EVALUATE if m is not None]
    
    print(f"\n{'Метрика':<25} {'Без реранкера':<20} {'С реранкером':<20} {'Изменение':<20}")
    print("-" * 85)
    
    for metric in metrics_names:
        no_rerank_val = results_no_rerank['mean'].get(metric, 0)
        with_rerank_val = results_with_rerank['mean'].get(metric, 0)
        diff = with_rerank_val - no_rerank_val
        diff_pct = (diff / no_rerank_val * 100) if no_rerank_val > 0 else 0
        diff_str = f"{diff:+.3f} ({diff_pct:+.1f}%)"
        
        print(f"{metric:<25} {no_rerank_val:<20.3f} {with_rerank_val:<20.3f} {diff_str:<20}")
    
    print("="*60)
    
    # Сохраняем результаты
    output = {
        'without_reranker': results_no_rerank,
        'with_reranker': results_with_rerank,
        'comparison': {
            metric: {
                'without': results_no_rerank['mean'].get(metric, 0),
                'with': results_with_rerank['mean'].get(metric, 0),
                'improvement': results_with_rerank['mean'].get(metric, 0) - results_no_rerank['mean'].get(metric, 0)
            }
            for metric in metrics_names
        }
    }
    
    output_path = "validation/ragas_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\nРезультаты сохранены в {output_path}")
    
    # Выводим таблицы в формате LaTeX (как в чекпоинте)
    print("\n" + "="*60)
    print("ТАБЛИЦЫ ДЛЯ ОТЧЕТА (LaTeX формат)")
    print("="*60)
    
    # Таблица без реранкера
    print("\n\\begin{table}[H]")
    print("\\centering")
    print("\\begin{tabular}{l" + "c" * len(metrics_names) + "}")
    print("\\toprule")
    print("& " + " & ".join(metrics_names) + " \\\\")
    print("\\midrule")
    for stat in ['mean', 'median', 'std', 'min', 'max']:
        print(f"{stat} & " + " & ".join([f"{results_no_rerank[stat].get(m, 0):.3f}" for m in metrics_names]) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Сводные метрики качества RAG-системы (без реранкера)}")
    print("\\end{table}\n")
    
    # Таблица с реранкером
    print("\n\\begin{table}[H]")
    print("\\centering")
    print("\\begin{tabular}{l" + "c" * len(metrics_names) + "}")
    print("\\toprule")
    print("& " + " & ".join(metrics_names) + " \\\\")
    print("\\midrule")
    for stat in ['mean', 'median', 'std', 'min', 'max']:
        print(f"{stat} & " + " & ".join([f"{results_with_rerank[stat].get(m, 0):.3f}" for m in metrics_names]) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Сводные метрики качества RAG-системы (с реранкером)}")
    print("\\end{table}\n")
    
    # Таблица сравнения
    print("\n\\begin{table}[H]")
    print("\\centering")
    print("\\begin{tabular}{l" + "c" * len(metrics_names) + "}")
    print("\\toprule")
    print("& " + " & ".join(metrics_names) + " \\\\")
    print("\\midrule")
    print("Без реранкера & " + " & ".join([f"{results_no_rerank['mean'].get(m, 0):.3f}" for m in metrics_names]) + " \\\\")
    print("С реранкером & " + " & ".join([f"{results_with_rerank['mean'].get(m, 0):.3f}" for m in metrics_names]) + " \\\\")
    print("Изменение & " + " & ".join([f"{results_with_rerank['mean'].get(m, 0) - results_no_rerank['mean'].get(m, 0):+.3f}" for m in metrics_names]) + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Сравнение средних метрик RAGAS}")
    print("\\end{table}\n")


if __name__ == "__main__":
    main()

