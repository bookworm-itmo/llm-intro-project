# Спецификация архитектуры RAG-системы

## Обзор

RAG (Retrieval-Augmented Generation) чат-бот для ответов на вопросы по роману "Мастер и Маргарита". Система находит релевантные фрагменты текста и генерирует ответы на их основе.

## Ключевые компоненты

### 1. Data Service

**Файлы:** `services/data_service/data_processor.py`, `services/data_service/data_preparator.py`

**Назначение:** Подготовка данных для RAG-системы.

**Классы:**
- `BookProcessor` - парсинг FB2, разбиение на главы и чанки
- `DataPreparator` - оркестрация полного пайплайна подготовки

**Ключевые параметры:**
| Параметр | Значение | Описание |
|----------|----------|----------|
| chunk_size | 800 | Размер чанка в символах |
| overlap | 100 | Перекрытие между чанками |
| min_chunk_length | 50 | Минимальная длина чанка |

**Алгоритм chunking:**
1. Парсинг FB2 через `xml.etree.ElementTree`
2. Разбиение по паттерну `Глава \d+`
3. `RecursiveCharacterTextSplitter` с разделителями: `["\n\n", "\n", ".", "!", "?", " "]`

### 2. RAG Service

**Файл:** `services/rag_service/rag_engine.py`

**Назначение:** Векторизация и семантический поиск.

**Класс:** `RAGEngine`

**Методы:**
| Метод | Описание |
|-------|----------|
| `build_chunks()` | Сохранение чанков в Parquet |
| `build_embeddings()` | Создание эмбеддингов через GigaChat |
| `build_faiss_index()` | Построение FAISS индекса |
| `search(query, top_k)` | Поиск релевантных чанков |
| `get_context_for_llm()` | Формирование контекста для LLM |

**Embedding pipeline:**
1. Запрос к GigaChat API (`embed_query` / `embed_documents`)
2. L2-нормализация векторов
3. Индексация в FAISS IndexFlatIP

**Параметры поиска:**
| Параметр | Значение |
|----------|----------|
| Размерность | 1024 |
| top_k (default) | 5 |
| Метрика | Inner Product (косинусное сходство) |

### 3. LLM Service

**Файл:** `services/llm_service/claude_client.py`

**Назначение:** Генерация ответов на основе контекста.

**Класс:** `ClaudeClient`

**Модель:** `claude-haiku-4-5-20251001`

**Промпт-инжиниринг:**
```
Ты - система для ответов на вопросы по роману "Мастер и Маргарита".

ВАЖНО: Ты должен отвечать ТОЛЬКО на основе предоставленного контекста.
НЕ используй свои знания о романе. Если ответа нет в контексте - так и скажи.

Контекст из книги:
{context}

Вопрос: {query}

Требования к ответу:
1. Используй ТОЛЬКО информацию из контекста выше
2. НЕ добавляй факты из своих знаний о романе
3. Если в контексте нет ответа, напиши: "В предоставленных фрагментах нет информации..."
4. Укажи главы источников
5. Приведи краткую цитату из контекста
```

### 4. Frontend

**Файл:** `frontend/app.py`

**Назначение:** Веб-интерфейс для взаимодействия с пользователем.

**Технология:** Streamlit

**Функции:**
- `load_rag_engine()` - кэшированная загрузка RAG-движка
- `load_llm_client()` - кэшированная инициализация LLM-клиента
- `main()` - основной UI с историей чата

## Хранилища данных

### Parquet-файлы

**chunks.parquet:**
```
chunk_id: int64      - уникальный ID чанка
chapter: int         - номер главы (1-32)
text: string         - текст фрагмента
```

**embeddings.parquet:**
```
chunk_id: int64      - ID чанка (FK к chunks)
embedding: array[float32]  - вектор размерности 1024
```

### FAISS Index

**Тип:** `IndexFlatIP` (Flat Inner Product)

**Характеристики:**
- Точный поиск (brute-force)
- Оптимизирован для косинусного сходства (с L2-нормализацией)
- Размер: 1182 вектора x 1024 измерения

## Внешние зависимости

### GigaChat API (Сбер)

**Использование:** Создание эмбеддингов

**Класс:** `langchain_gigachat.embeddings.GigaChatEmbeddings`

**Параметры:**
```python
credentials=GIGACHAT_AUTH_KEY
scope="GIGACHAT_API_PERS"
verify_ssl_certs=False
```

### Claude API (Anthropic)

**Использование:** Генерация ответов

**Класс:** `anthropic.Anthropic`

**Модель:** `claude-haiku-4-5-20251001`

**Параметры:**
```python
max_tokens=1024
```

## Потоки данных

### Offline: Подготовка данных

```
FB2 File
    |
    v
BookProcessor.load_book()
    |
    v
BookProcessor.split_into_chapters()
    |
    v
BookProcessor.create_chunks()
    |
    v
RAGEngine.build_chunks() --> chunks.parquet
    |
    v
RAGEngine.build_embeddings() --> GigaChat API --> embeddings.parquet
    |
    v
RAGEngine.build_faiss_index() --> index.faiss
```

### Online: Обработка запроса

```
User Question
    |
    v
Frontend (Streamlit)
    |
    v
RAGEngine.get_context_for_llm()
    |
    +---> GigaChat API (embed_query)
    |
    +---> FAISS (search)
    |
    +---> chunks.parquet (get text)
    |
    v
ClaudeClient.generate_answer()
    |
    +---> Claude API (messages.create)
    |
    v
Response + Sources --> User
```

## Метрики качества

### Baseline (Claude Haiku 4.5 + GigaChat Embedding)

| Метрика | Значение | Интерпретация |
|---------|----------|---------------|
| faithfulness | 0.755 | Высокий - мало галлюцинаций |
| answer_relevancy | 0.418 | Низкий - много отказов |
| context_recall | 0.642 | Средний - бимодальное распределение |
| context_precision | 0.360 | Низкий - шум в контексте |
| context_utilization | 0.382 | Низкий - модель не использует контекст |

### Выявленные проблемы

1. **Retrieval gaps:** 23 из 70 запросов с context_recall < 0.5
2. **Избыточная осторожность:** 15 отказов при наличии ответа в контексте
3. **Низкая утилизация:** модель не извлекает информацию из найденного контекста

## Ограничения

1. **Точный поиск FAISS:** O(n) сложность, не масштабируется на большие корпуса
2. **Один источник:** только роман "Мастер и Маргарита"
3. **Без истории:** каждый запрос независим, нет multi-turn диалога
4. **Без reranking:** используются первые top-k результатов без переранжирования
