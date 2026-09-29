# ADR-004: Стратегия sparse/hybrid search для GraphRAG-системы

**Статус:** Proposed  
**Дата:** 2026-04-24  
**Автор:** AI-Architect  

## Контекст

Dense retrieval из ADR-003 хорошо покрывает семантические запросы, но GraphRAG по научным статьям требует точного поиска по именованным сущностям и табличным результатам:

- модели и методы: `GPT-2`, `LLaMA`, `S2IP-LLM`, `One-for-All`, `rsLoRA`;
- датасеты: `ETT`, `ETTm1`, `Weather`, `Traffic`;
- метрики: `MSE`, `MAE`, `sMAPE`, `CRPS`;
- разделы и типы чанков: `Methodology`, `Appendix_Results`, `table`, `formula_heavy`;
- численные результаты и обозначения горизонтов прогнозирования.

Часть этих сигналов уже есть в payload (`keywords`, `section`, `section_hierarchy`, `chunk_type`), но `keywords` являются предварительным приближением entity extraction и не должны быть единственным lexical-механизмом. Нужен sparse-слой, который ищет по фактическому тексту чанка и дополняет dense retrieval.

ADR-002 выбрал Qdrant. Qdrant поддерживает named vectors, sparse vectors и Query API для multi-stage/hybrid retrieval, поэтому sparse/hybrid search можно реализовать в той же коллекции.

## Решение

**Использовать Qdrant-native hybrid search:** объединять dense-вектор `dense_retrieval` и sparse-вектор `sparse_bm25` через Reciprocal Rank Fusion (`RRF`).

Основная схема:

| Слой | Vector name | Источник | Назначение |
| --- | --- | --- | --- |
| Dense | `dense_retrieval` | `retrieval_text` | Семантическая близость, парафразы, обобщённые вопросы |
| Sparse | `sparse_bm25` | `sparse_text` | Точные термины, названия моделей, датасеты, метрики, числа |
| Rerank | внешний reranker / LLM | полный `text` | Финальное упорядочивание и проверка контекста |

Sparse-представление на первом этапе строится как BM25 sparse vector. SPLADE/miniCOIL и другие neural sparse retrievers откладываются до появления evaluation-набора и необходимости улучшать качество сверх BM25.

## Детали реализации

### Поле `sparse_text`

Для sparse indexing создаётся отдельное поле `sparse_text`.

```python
section_path = " > ".join(chunk["section_hierarchy"])
sparse_text = "\n".join([
    f"Title: {chunk['article_meta']['title']}",
    f"Section: {section_path}",
    f"Summary: {chunk['summary']}",
    f"Keywords: {', '.join(chunk['keywords'])}",
    f"Text: {chunk['text']}",
])
```

`sparse_text` хранится в payload. Это нужно для воспроизводимости sparse-вектора и отладки lexical retrieval.

Причина включения полного `text`: точные обозначения моделей, датасетов, метрик, формул и численных результатов часто находятся в таблицах и appendix-разделах и могут отсутствовать в `summary`.

### Конфигурация Qdrant

Коллекция `chunks` должна поддерживать оба представления:

```python
client.create_collection(
    collection_name="chunks",
    vectors_config={
        "dense_retrieval": models.VectorParams(
            size=3072,
            distance=models.Distance.COSINE,
        ),
    },
    sparse_vectors_config={
        "sparse_bm25": models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        ),
    },
)
```

Sparse-векторы генерируются через Qdrant BM25 inference или локально через `qdrant-client[fastembed]`. Для воспроизводимости в metadata загрузки фиксируются:

```json
{
  "sparse_vector_name": "sparse_bm25",
  "sparse_model": "Qdrant/bm25",
  "sparse_input_field": "sparse_text",
  "sparse_modifier": "IDF",
  "hybrid_fusion": "RRF"
}
```

### Query pipeline

Базовый retrieval-профиль:

1. Из пользовательского запроса строится dense embedding через `text-embedding-3-large`.
2. Из того же запроса строится BM25 sparse vector.
3. В Qdrant выполняются две prefetch-ветки:
   - `dense_retrieval`, например `limit=80`;
   - `sparse_bm25`, например `limit=80`.
4. Результаты объединяются через `RRF`.
5. Возвращается `top-30` кандидатов с payload.
6. `top-10` или `top-15` передаются в reranker / LLM с полным `text`.
7. При необходимости контекст расширяется соседними чанками через `article_id + index`.

Фильтры по `section`, `chunk_type`, `article_meta.year`, `article_id`, `keywords` применяются нативно в Qdrant и должны передаваться в обе prefetch-ветки. Постфильтрация допускается только как исключение с oversampling.

Пример формы запроса:

```python
client.query_points(
    collection_name="chunks",
    prefetch=[
        models.Prefetch(
            query=dense_query_vector,
            using="dense_retrieval",
            limit=80,
            filter=query_filter,
        ),
        models.Prefetch(
            query=sparse_query_vector,
            using="sparse_bm25",
            limit=80,
            filter=query_filter,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
    limit=30,
    with_payload=True,
)
```

### Профили поиска

| Профиль | Когда использовать | Настройка |
| --- | --- | --- |
| `hybrid_default` | Обычные вопросы пользователя | Dense `80` + sparse `80`, RRF, final `30` |
| `dense_only` | Диагностика и semantic baseline | Только `dense_retrieval` |
| `sparse_only` | Диагностика lexical baseline | Только `sparse_bm25` |
| `entity_exact` | Запрос явно содержит модель, датасет, метрику или arXiv ID | Hybrid + metadata filter по распознанным `keywords`/`article_meta`, если сущность уверенно нормализована |
| `table_numeric` | Запросы о численных результатах, метриках, сравнениях | Hybrid + приоритет `chunk_type in ["table", "table_with_prose", "formula_heavy"]` |

`hybrid_default` является основным профилем после готовности sparse-векторов. `dense_only` сохраняется как fallback, если sparse-индексация временно недоступна.

## Причины выбора

1. **Dense и sparse закрывают разные ошибки.** Dense лучше находит парафразы, sparse лучше ловит точные обозначения моделей, датасетов, метрик и чисел.
2. **BM25 прозрачен и устойчив для технических терминов.** Для научных статей важны точные токены, а не только нейросемантическое расширение запроса.
3. **RRF не требует сопоставлять разные шкалы score.** Cosine-score dense-поиска и BM25-score sparse-поиска имеют разную природу, поэтому rank-based fusion является безопасным baseline.
4. **Одна база данных.** Qdrant позволяет хранить dense и sparse vectors в одной коллекции и выполнять гибридный запрос через Query API.
5. **Хорошая отладка.** Можно сравнивать dense-only, sparse-only и hybrid выдачи на одном payload и одном наборе фильтров.

## Рассмотренные альтернативы

| Альтернатива | Причина отклонения |
| --- | --- |
| Только dense + payload filters | Недостаточно для точных терминов, чисел и таблиц. `keywords` неполны и не заменяют lexical search по тексту. |
| Отдельный Elasticsearch / OpenSearch для BM25 | Мощное решение, но добавляет второй retrieval backend и риск рассинхронизации с Qdrant. Для текущего масштаба Qdrant sparse vectors проще. |
| SPLADE как первый sparse retriever | Может дать лучшее качество, но тяжелее, требует отдельной оценки и внимательного выбора лицензии/модели. BM25 лучше подходит как прозрачный baseline. |
| Только metadata keyword search | Работает для уже извлечённых `keywords`, но не покрывает пропущенные сущности, таблицы, формулы и численные результаты в `text`. |
| Score-weighted dense+sparse fusion | Требует калибровки score между разными retrieval-сигналами. На первом этапе RRF проще и устойчивее. |

## Evaluation

Перед фиксацией `hybrid_default` как продуктивного профиля нужен небольшой retrieval-evaluation набор:

- 30-50 запросов по онтологии: Method, Task, BaseLLM, Technique, Dataset, Metric, ExperimentResult;
- отдельные запросы по таблицам и appendix-разделам;
- запросы с точными токенами (`GPT-2`, `ETTm1`, `MSE`, `rsLoRA`);
- запросы с парафразами без точных терминов.

Сравниваются `dense_only`, `sparse_only`, `hybrid_default`:

- `Recall@20`;
- `MRR@10`;
- `nDCG@10`;
- ручная проверка top-10 для сложных научных вопросов.

Гибридный профиль считается успешным, если он повышает recall по точным entity/table-запросам и не даёт заметной деградации по чисто семантическим запросам.

## Последствия

### Положительные

- Повышается recall по точным названиям моделей, датасетов, метрик и техник.
- Улучшается поиск по таблицам, appendix-results и численным сравнениям.
- Qdrant остаётся единственным retrieval backend.
- Сохраняется возможность диагностировать dense и sparse ветки отдельно.

### Риски и требования

- Индексация усложняется: нужно строить и хранить два представления чанка.
- При изменении `sparse_text`, BM25-параметров или sparse-модели нужна переиндексация sparse-векторов.
- BM25 зависит от токенизации; для технических обозначений нужно проверить обработку дефисов, регистра и чисел.
- RRF улучшает устойчивость, но не заменяет reranking по полному `text`.
- Для точных численных запросов может потребоваться отдельный структурный индекс experiment results после построения графа знаний.

## Источники

- Qdrant Hybrid Queries: Query API, prefetch, RRF, dense+sparse fusion: https://qdrant.tech/documentation/search/hybrid-queries/
- Qdrant Text Search: BM25 sparse vectors и FastEmbed/Qdrant inference: https://qdrant.tech/documentation/guides/text-search/
- ADR-002 — выбор Qdrant как векторной базы данных.
- ADR-003 — выбор dense embedding-модели и размерности.
