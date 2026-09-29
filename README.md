# GraphRAG по статьям о прогнозировании временных рядов

Проект индексирует фрагменты научных статей, ищет доказательные фрагменты через sparse, dense и hybrid retrieval, связывает статьи, чанки и сущности в Neo4j и формирует ответ по найденному контексту. В демонстрационном корпусе `chunks 2/docs/dic_chunks.json` — 141 чанк из 12 статей. В этом же репозитории находится модуль подготовки Markdown `ts-llm-chunker/`.

## Быстрый запуск без сервисов и API ключей

Нужен Python 3.12+; команды выполняются из этой папки.

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .[dev]
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\vector-db-rag validate-chunks
.\.venv\Scripts\vector-db-rag local-query "Time-LLM reprogramming text prototypes" --limit 3
.\.venv\Scripts\vector-db-rag offline-answer "How does Time-LLM use text prototypes?" --limit 3
.\.venv\Scripts\vector-db-rag offline-eval
```

`local-query` ранжирует чанки локальным BM25. `offline-answer` расширяет результаты по графу `Chunk → Keyword → Chunk` и копирует доказательные предложения из исходных чанков с `chunk_id`. Эти команды не используют Qdrant, Neo4j или LLM. `offline-eval` считает hit@1, hit@3 и проверяет, что цитаты дословно встречаются в корпусе на шести фиксированных вопросах; это демонстрационная проверка, а не независимый benchmark.

## Связь с чанкером

Модуль `ts-llm-chunker/` находится в этом репозитории. После его запуска можно преобразовать базовый выход в минимальный корпус для поиска:

```powershell
Push-Location ts-llm-chunker
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .[dev]
.\.venv\Scripts\ts-build --input articles_raw --output build
Pop-Location
.\.venv\Scripts\vector-db-rag adapt-chunker --chunks ts-llm-chunker\build\chunks\dic_chunks.json --articles ts-llm-chunker\build\articles --output data\basic_chunks.json
.\.venv\Scripts\vector-db-rag validate-chunks data\basic_chunks.json
.\.venv\Scripts\vector-db-rag local-query "S2IP-LLM semantic anchors" --chunks data\basic_chunks.json --limit 3
```

Адаптер копирует текст и метаданные из Markdown, строит короткое извлечение из текста в поле `summary` и оставляет семантические теги пустыми. Полученный корпус пригоден для демонстрации поиска, но не заменяет вручную обогащённый корпус из 12 статей.

## Полный контур

```powershell
Copy-Item .env.example .env
docker compose up -d
.\.venv\Scripts\vector-db-rag create-collection
.\.venv\Scripts\vector-db-rag index-chunks --sparse
.\.venv\Scripts\vector-db-rag create-graph-schema
.\.venv\Scripts\vector-db-rag index-graph
.\.venv\Scripts\vector-db-rag query "Time-LLM reprogramming text prototypes" --profile sparse_only --limit 3
.\.venv\Scripts\vector-db-rag answer "Как Time-LLM использует text prototypes?" --profile sparse_only --dry-run --json --no-graph
```

Для dense или hybrid поиска нужен `OPENAI_API_KEY` в `.env` и индексация `--dense`. Для ответа через модель нужен `DEEPSEEK_API_KEY`; `--dry-run` показывает подобранные источники и промпт без сетевого вызова. Флаг `--no-graph` позволяет запускать ответ до импорта в Neo4j.

## Данные и граница автоматизации

`ts-llm-chunker/` автоматически очищает Markdown и создаёт базовые чанки из двух примеров. Основной контур по умолчанию читает другой, обогащённый корпус v1.2.0: у каждого чанка есть метаданные статьи, иерархия разделов, тип, summary, keywords и question_types. Адаптер `adapt-chunker` строит совместимый минимальный JSON, но не создаёт предметные keywords и question_types. Проверка обеих схем доступна через `validate-chunks` и тесты.

Схема: Markdown → базовые чанки → семантическое обогащение → корпус v1.2.0 → Qdrant/Neo4j → поиск → ответ с источниками.

Подробные архитектурные решения и примеры команд находятся в `doc/`. Для защиты используйте текущий валидированный корпус и отдельно называйте ручной этап его подготовки.

Исходные проекты до объединения находились в отдельных репозиториях `nullnesse/graph-rag` и `nullnesse/ts-llm-chunker`; история их коммитов сохранена. Изменения в этом репозитории добавляют единый сборочный путь, адаптер схем, локальный GraphRAG, проверку качества и исправления индексирования.
