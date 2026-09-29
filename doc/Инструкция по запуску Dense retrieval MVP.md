# Инструкция по запуску Dense и Hybrid retrieval MVP

Документ описывает запуск Milestone 1-2: валидация чанков, локальный Qdrant, создание коллекции, dense indexing, sparse BM25 indexing, dense-only retrieval и hybrid retrieval.

## Установка зависимостей

Из корня проекта:

```powershell
pip install -e .
```

Если пакет не установлен, команды можно запускать через `PYTHONPATH`:

```powershell
$env:PYTHONPATH = "src"
python -m vector_db_rag validate-chunks
```

## Переменные окружения

Создать `.env` на основе `.env.example`:

```powershell
Copy-Item .env.example .env
```

Заполнить:

```text
OPENAI_API_KEY=...
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
```

Для локального Qdrant `QDRANT_API_KEY` обычно остаётся пустым.

## Запуск Qdrant

```powershell
docker compose up -d qdrant
```

Проверка подключения:

```powershell
vector-db-rag check-qdrant
```

или без установки пакета:

```powershell
$env:PYTHONPATH = "src"
python -m vector_db_rag check-qdrant
```

## Валидация чанков

```powershell
vector-db-rag validate-chunks
```

Ожидаемый результат для текущего корпуса:

- `chunks`: `163`;
- `articles`: `2`;
- `chunker_version`: `0.1.0`;
- `config_hash`: `2f09dc9dfd28`.

## Создание коллекции

```powershell
vector-db-rag create-collection
```

Пересоздание коллекции:

```powershell
vector-db-rag create-collection --recreate --yes
```

Команда создаёт коллекцию `chunks` с:

- dense named vector `dense_retrieval`, размерность `3072`, distance `Cosine`;
- sparse vector placeholder `sparse_bm25` для Milestone 2;
- payload indexes по полям ADR-002.

## Dense indexing

```powershell
vector-db-rag index-chunks --dense
```

Команда:

1. Читает `chunks/dic_chunks.json`.
2. Строит `retrieval_text`.
3. Получает embedding через OpenAI `text-embedding-3-large`.
4. Загружает точки в Qdrant.
5. Пишет manifest в `data/index_manifest.json`.
6. Кэширует embeddings в `data/cache/embeddings`.

## Sparse BM25 indexing

После dense indexing можно добавить sparse-векторы:

```powershell
vector-db-rag index-chunks --sparse
```

Команда:

1. Строит `sparse_text` для каждого чанка.
2. Обучает локальный BM25 encoder на текущем корпусе.
3. Сохраняет BM25 metadata в `data/cache/sparse_bm25.json`.
4. Обновляет Qdrant sparse vector `sparse_bm25` без перезаписи dense-вектора.
5. Добавляет в payload `sparse_text` и sparse metadata.

Полная переиндексация dense + sparse:

```powershell
vector-db-rag create-collection --recreate --yes
vector-db-rag index-chunks --dense --sparse
```

## Dense-only query

Пример без фильтров:

```powershell
vector-db-rag query "methods based on GPT-2 for time series forecasting" --limit 5
```

Пример с фильтрами:

```powershell
vector-db-rag query "methods based on GPT-2" --section Methodology --keyword GPT-2 --year-gte 2023 --limit 5
```

Поддерживаемые фильтры Milestone 1:

- `--article-id`;
- `--arxiv-id`;
- `--section`;
- `--chunk-type`;
- `--keyword`;
- `--year-gte`;
- `--year-lte`.

## Sparse-only query

```powershell
vector-db-rag query "GPT-2 MSE ETT" --profile sparse_only --limit 5
```

Профиль `sparse_only` не вызывает OpenAI API, но требует файл `data/cache/sparse_bm25.json`, созданный командой `index-chunks --sparse`.

## Hybrid query

Обычный гибридный поиск:

```powershell
vector-db-rag query "methods based on GPT-2 for time series forecasting" --profile hybrid_default --limit 10
```

Гибридный поиск с entity-фильтрами:

```powershell
vector-db-rag query "methods based on GPT-2" --profile entity_exact --keyword GPT-2 --section Methodology --limit 5
```

Профиль для таблиц и численных результатов:

```powershell
vector-db-rag query "MSE results on ETT datasets" --profile table_numeric --limit 10
```

`table_numeric` автоматически ограничивает `chunk_type` значениями `table`, `table_with_prose`, `formula_heavy`, если явно не передан `--chunk-type`.

## Локальные проверки

```powershell
python -m pytest
python -m compileall src tests
```
