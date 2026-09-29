# План поэтапной реализации ADR

**Статус:** Draft  
**Дата:** 2026-04-24  
**Область:** реализация retrieval-слоя GraphRAG по ADR-001, ADR-002, ADR-003, ADR-004  

## Цель

Построить воспроизводимый retrieval-слой для GraphRAG-системы по научным статьям arXiv на тему *LLM for Time Series Forecasting*:

1. Проверить и закрепить контракт чанков из `chunks/dic_chunks.json`.
2. Развернуть Qdrant как векторное хранилище.
3. Создать коллекцию `chunks` с dense и sparse представлениями.
4. Индексировать чанки с OpenAI `text-embedding-3-large` и BM25 sparse-векторами.
5. Реализовать dense-only, sparse-only и hybrid retrieval.
6. Подготовить evaluation-набор и сравнить профили поиска.
7. Зафиксировать операционный контур переиндексации и отладки.

## Покрываемые ADR

| ADR | Решение | Что должно быть реализовано |
| --- | --- | --- |
| ADR-001 | Стратегия чанкинга и структура обогащённого чанка | Валидация `dic_chunks.json`, проверка схемы, использование `article_id + index` для соседей |
| ADR-002 | Qdrant как векторная БД | Docker Compose, коллекция `chunks`, payload indexes, загрузка payload |
| ADR-003 | OpenAI `text-embedding-3-large`, `3072` | Построение `retrieval_text`, dense embedding, named vector `dense_retrieval`, manifest индексации |
| ADR-004 | BM25 sparse + RRF hybrid search | Построение `sparse_text`, sparse vector `sparse_bm25`, hybrid query через Qdrant Query API |

## Предпосылки

- Установлен Python 3.11+.
- Доступен Docker Desktop или совместимый Docker runtime.
- Есть `OPENAI_API_KEY` для построения dense embeddings.
- `chunks/dic_chunks.json` является входным источником данных.
- На первом этапе графовая БД и финальное извлечение сущностей/отношений не реализуются; retrieval-слой должен быть готов к их последующему подключению.

## Этап 0. Каркас проекта и конфигурация

**Цель:** создать минимальную техническую основу, чтобы последующие этапы были воспроизводимыми.

Работы:

1. Создать Python-пакет проекта, например `src/vector_db_rag`.
2. Добавить конфигурационные файлы:
   - `.env.example`;
   - `config/retrieval.yaml` или эквивалент;
   - `pyproject.toml` с зависимостями.
3. Зафиксировать базовые зависимости:
   - `qdrant-client`;
   - `openai`;
   - `pydantic`;
   - `typer` или `argparse` для CLI;
   - `pytest`;
   - `python-dotenv`;
   - `qdrant-client[fastembed]` или отдельный `fastembed`, если BM25 строится локально.
4. Подготовить CLI-команды верхнего уровня:
   - `validate-chunks`;
   - `create-collection`;
   - `index-chunks`;
   - `query`;
   - `evaluate`.

Артефакты:

- `pyproject.toml`;
- `.env.example`;
- `config/retrieval.yaml`;
- `src/vector_db_rag/...`;
- `tests/...`.

Критерии готовности:

- Команды CLI запускаются и выводят help.
- Конфигурация читается из файла и переменных окружения.
- Тестовая команда `pytest` стартует без инфраструктурных ошибок.

## Этап 1. Контракт данных и валидация чанков

**Цель:** убедиться, что входные данные соответствуют ADR-001 и пригодны для индексации.

Работы:

1. Описать Pydantic-модели:
   - `ChunkFile`;
   - `Chunk`;
   - `ArticleMeta`.
2. Реализовать загрузчик `chunks/dic_chunks.json`.
3. Проверить обязательные условия:
   - JSON валиден;
   - `chunks` является массивом;
   - каждый чанк содержит `chunk_id`, `article_id`, `article_meta`, `section`, `section_hierarchy`, `chunk_type`, `index`, `text`, `summary`, `keywords`, `token_count`;
   - `chunk_id` уникален;
   - пара `article_id + index` уникальна;
   - `section_hierarchy` не пуст;
   - `summary` и `text` не пустые;
   - `token_count > 0`;
   - `chunk_type` входит в допустимое множество.
4. Реализовать построение производных полей:
   - `retrieval_text`;
   - `sparse_text`.
5. Сформировать manifest индексации:
   - версия чанкинга;
   - `config_hash`;
   - модель embedding;
   - размерность;
   - имена vector-полей;
   - дата индексации;
   - количество чанков.

Артефакты:

- `src/vector_db_rag/chunks.py`;
- `src/vector_db_rag/text_fields.py`;
- `src/vector_db_rag/manifest.py`;
- тесты на валидатор и построение `retrieval_text` / `sparse_text`.

Критерии готовности:

- `validate-chunks chunks/dic_chunks.json` завершается успешно.
- Отчёт валидации показывает количество статей, чанков, секций и типов чанков.
- Производные поля строятся детерминированно и не записываются обратно в исходный `dic_chunks.json` без отдельного решения.

## Этап 2. Локальный Qdrant

**Цель:** поднять локальное векторное хранилище, совместимое с ADR-002.

Работы:

1. Добавить `docker-compose.yml` с Qdrant.
2. Зафиксировать volume для локальных данных Qdrant.
3. Добавить healthcheck или отдельную команду проверки доступности.
4. Реализовать клиент Qdrant:
   - чтение URL/API key из конфигурации;
   - проверка подключения;
   - получение информации о коллекции.

Артефакты:

- `docker-compose.yml`;
- `src/vector_db_rag/qdrant_client.py`;
- команда `check-qdrant`.

Критерии готовности:

- Qdrant запускается локально.
- Клиент успешно получает service/collection info.
- Ошибки подключения диагностируются понятным сообщением.

## Этап 3. Создание коллекции и payload indexes

**Цель:** создать коллекцию `chunks` с конфигурацией ADR-002, ADR-003 и ADR-004.

Работы:

1. Создать коллекцию `chunks`:
   - dense named vector `dense_retrieval`;
   - размерность `3072`;
   - distance `Cosine`;
   - sparse vector `sparse_bm25`;
   - `IDF` modifier для BM25 sparse-вектора.
2. Создать payload indexes:
   - `chunk_id`: keyword;
   - `article_id`: keyword;
   - `article_meta.year`: integer;
   - `article_meta.arxiv_id`: keyword;
   - `section`: keyword;
   - `section_hierarchy`: keyword;
   - `chunk_type`: keyword;
   - `keywords`: keyword;
   - `index`: integer.
3. Реализовать режимы:
   - `create`;
   - `recreate`;
   - `ensure`.
4. Добавить защиту от случайного удаления коллекции:
   - `recreate` требует явный флаг `--yes`.

Артефакты:

- `src/vector_db_rag/collection.py`;
- команда `create-collection`;
- тесты конфигурации коллекции, где возможно без живого Qdrant через unit-level проверку параметров.

Критерии готовности:

- Коллекция создаётся одной командой.
- Повторный запуск в режиме `ensure` идемпотентен.
- Payload indexes видны в Qdrant.

## Этап 4. Dense indexing по ADR-003

**Цель:** построить и загрузить dense-векторы `dense_retrieval`.

Работы:

1. Реализовать OpenAI embedding-клиент:
   - модель `text-embedding-3-large`;
   - размерность по умолчанию `3072`;
   - batch processing;
   - retries с backoff;
   - контроль пустых входов.
2. Для каждого чанка построить `retrieval_text`.
3. Сформировать payload:
   - исходные поля чанка;
   - `retrieval_text`;
   - служебные поля `chunker_version`, `config_hash`;
   - `embedding_vector_name`;
   - `embedding_provider`;
   - `embedding_model`;
   - `embedding_dim`;
   - `embedding_input_field`.
4. Загрузить точки в Qdrant batch-ами.
5. Добавить локальный cache embeddings, чтобы повторная индексация не вызывала API без необходимости.

Артефакты:

- `src/vector_db_rag/embeddings.py`;
- `src/vector_db_rag/index_dense.py`;
- `data/cache/embeddings/...` или иной явно зафиксированный cache-каталог;
- команда `index-chunks --dense`.

Критерии готовности:

- Все чанки загружены в Qdrant с dense-вектором.
- Размерность каждого dense-вектора равна `3072`.
- Payload содержит `retrieval_text` и metadata индексации.
- Повторный запуск не создаёт дубликаты и не смешивает модели.

## Этап 5. Dense-only retrieval baseline

**Цель:** получить первый рабочий поиск до включения sparse/hybrid.

Работы:

1. Реализовать кодирование пользовательского запроса через `text-embedding-3-large`.
2. Реализовать `dense_only` query profile:
   - vector name `dense_retrieval`;
   - configurable `limit`;
   - `with_payload=True`;
   - поддержка `query_filter`.
3. Реализовать сборку Qdrant-фильтров:
   - год;
   - статья;
   - arXiv ID;
   - section;
   - chunk_type;
   - keywords.
4. Добавить форматированный вывод результатов:
   - score;
   - `chunk_id`;
   - title/year;
   - `section_hierarchy`;
   - краткий preview `summary`.
5. Добавить восстановление соседних чанков через `article_id + index`.

Артефакты:

- `src/vector_db_rag/filters.py`;
- `src/vector_db_rag/retrieval.py`;
- команда `query --profile dense_only`.

Критерии готовности:

- Можно выполнить запрос вида `"methods based on GPT-2 for time series forecasting"`.
- Можно ограничить поиск фильтром `section=Methodology`.
- Результаты содержат payload, достаточный для ручной проверки.

## Этап 6. Sparse indexing по ADR-004

**Цель:** добавить lexical retrieval по `sparse_text`.

Работы:

1. Реализовать построение `sparse_text` для каждого чанка.
2. Выбрать технический режим BM25:
   - предпочтительно Qdrant inference, если доступен в выбранной среде;
   - fallback: локальная генерация через `qdrant-client[fastembed]`.
3. Построить sparse-векторы для `sparse_bm25`.
4. Обновить точки в Qdrant без потери dense-векторов и payload.
5. Зафиксировать sparse metadata:
   - `sparse_vector_name`;
   - `sparse_model`;
   - `sparse_input_field`;
   - `sparse_modifier`;
   - `hybrid_fusion`.

Артефакты:

- `src/vector_db_rag/sparse.py`;
- `src/vector_db_rag/index_sparse.py`;
- команда `index-chunks --sparse`.

Критерии готовности:

- Все точки имеют sparse-вектор `sparse_bm25`.
- `sparse_text` хранится в payload или воспроизводимо строится из payload.
- `sparse_only` поиск находит чанки по точным терминам `GPT-2`, `MSE`, `ETT`, `rsLoRA`.

## Этап 7. Hybrid retrieval и профили поиска

**Цель:** реализовать основной retrieval-профиль ADR-004.

Работы:

1. Реализовать `sparse_only` query profile.
2. Реализовать `hybrid_default`:
   - dense prefetch `limit=80`;
   - sparse prefetch `limit=80`;
   - fusion `RRF`;
   - final `limit=30`.
3. Реализовать специализированные профили:
   - `entity_exact`;
   - `table_numeric`.
4. Передавать одинаковый `query_filter` в dense и sparse prefetch-ветки.
5. Добавить oversampling только для фильтров, которые нельзя выразить в Qdrant.
6. Сохранять диагностическую информацию:
   - profile;
   - filters;
   - dense/sparse/hybrid result ids;
   - итоговый ranking.

Артефакты:

- расширение `src/vector_db_rag/retrieval.py`;
- `src/vector_db_rag/profiles.py`;
- команда `query --profile hybrid_default`.

Критерии готовности:

- `hybrid_default` возвращает результаты через один Qdrant Query API вызов.
- `dense_only`, `sparse_only`, `hybrid_default` доступны как независимые профили.
- Фильтрация по metadata работает одинаково для dense и hybrid.

## Этап 8. Reranking и сбор контекста

**Цель:** подготовить кандидаты retrieval для последующей генерации ответа и GraphRAG-извлечения.

Работы:

1. Реализовать интерфейс reranker:
   - сначала no-op / score passthrough;
   - затем cross-encoder или LLM-as-reranker отдельной конфигурацией.
2. На вход reranker подавать полный `text`, а не `retrieval_text`.
3. Реализовать context expansion:
   - `same article, index - 1`;
   - `same article, index + 1`;
   - ограничение по суммарному `token_count`.
4. Реализовать дедупликацию соседних чанков.
5. Подготовить структуру результата для следующего слоя GraphRAG:
   - найденный чанк;
   - соседние чанки;
   - source metadata;
   - ranking explanation.

Артефакты:

- `src/vector_db_rag/rerank.py`;
- `src/vector_db_rag/context.py`;
- команда `query --rerank --expand-context`.

Критерии готовности:

- Поиск возвращает top-K с полным `text`.
- Можно включить соседние чанки без нарушения порядка внутри статьи.
- Reranker можно заменить без изменения Qdrant-слоя.

## Этап 9. Retrieval evaluation

**Цель:** проверить, что выбранные ADR дают измеримое качество retrieval.

Работы:

1. Создать evaluation-набор `eval/retrieval_queries.yaml`.
2. Покрыть 30-50 запросов по классам онтологии:
   - `Method`;
   - `Task`;
   - `BaseLLM`;
   - `Technique`;
   - `Dataset`;
   - `Metric`;
   - `ExperimentResult`;
   - таблицы и appendix-разделы.
3. Для каждого запроса зафиксировать:
   - текст запроса;
   - ожидаемые `chunk_id` или `article_id + section`;
   - тип запроса: semantic, exact, table_numeric, filtered.
4. Реализовать метрики:
   - `Recall@20`;
   - `MRR@10`;
   - `nDCG@10`;
   - ручная выгрузка top-10.
5. Сравнить профили:
   - `dense_only`;
   - `sparse_only`;
   - `hybrid_default`;
   - `table_numeric`, где применимо.

Артефакты:

- `eval/retrieval_queries.yaml`;
- `src/vector_db_rag/evaluate.py`;
- `reports/retrieval_eval.md`;
- machine-readable отчёт `reports/retrieval_eval.json`.

Критерии готовности:

- Есть baseline-таблица качества по всем профилям.
- `hybrid_default` не хуже `dense_only` на семантических запросах в пределах согласованного допуска.
- `hybrid_default` лучше `dense_only` на exact/table-запросах.
- Найдены и задокументированы слабые места: токенизация BM25, пропущенные keywords, плохие summary, проблемные таблицы.

## Этап 10. Операционный контур переиндексации

**Цель:** сделать индексацию повторяемой и безопасной.

Работы:

1. Реализовать команду полной переиндексации:
   - validate;
   - recreate collection;
   - create indexes;
   - dense indexing;
   - sparse indexing;
   - smoke queries;
   - evaluation.
2. Добавить manifest в Qdrant или рядом с индексом:
   - `index_manifest.json`;
   - версия ADR/config;
   - дата и параметры индексации;
   - количество точек.
3. Добавить smoke-test запросы:
   - `"GPT-2"`;
   - `"MSE on ETT"`;
   - `"decomposition method"`;
   - `"Appendix results forecasting horizon"`.
4. Добавить проверки против смешивания индексов:
   - несовпадение `embedding_model`;
   - несовпадение `embedding_dim`;
   - несовпадение `chunker_version` / `config_hash`.

Артефакты:

- команда `reindex`;
- `index_manifest.json`;
- smoke-test сценарий;
- документация запуска.

Критерии готовности:

- Новый индекс можно пересобрать одной командой.
- Ошибочная попытка загрузить векторы другой модели блокируется.
- Smoke queries проходят после переиндексации.

## Этап 11. Документация для разработчика

**Цель:** сделать систему понятной для продолжения разработки.

Работы:

1. Подготовить `README.md` или `doc/Инструкция по запуску retrieval-слоя.md`.
2. Описать:
   - запуск Qdrant;
   - настройку `OPENAI_API_KEY`;
   - создание коллекции;
   - индексацию;
   - query examples;
   - evaluation;
   - переиндексацию.
3. Добавить troubleshooting:
   - Qdrant недоступен;
   - OpenAI API key отсутствует;
   - размерность вектора не совпадает;
   - sparse-векторы не построены;
   - фильтр не возвращает результатов.

Артефакты:

- `README.md` или отдельная инструкция в `doc`;
- примеры команд.

Критерии готовности:

- Новый разработчик может поднять Qdrant, загрузить чанки и выполнить query по инструкции.
- Основные ошибки имеют понятные способы диагностики.

## Рекомендуемая последовательность релизов

### Milestone 1. Dense retrieval MVP

Включает этапы 0-5.

Результат: локальный Qdrant, валидированный корпус, dense-векторы `text-embedding-3-large`, поиск `dense_only` с payload-фильтрами.

### Milestone 2. Hybrid retrieval

Включает этапы 6-7.

Результат: sparse BM25-векторы, `sparse_only`, `hybrid_default`, специализированные профили `entity_exact` и `table_numeric`.

### Milestone 3. Retrieval quality loop

Включает этапы 8-9.

Результат: reranking-интерфейс, расширение контекста соседними чанками, evaluation-набор и отчёт сравнения профилей.

### Milestone 4. Reproducible indexing

Включает этапы 10-11.

Результат: полная переиндексация одной командой, manifest, smoke-tests, инструкция запуска.

## Приоритеты реализации

1. Сначала валидировать данные и создать Qdrant-коллекцию.
2. Затем получить dense-only baseline, потому что он быстрее всего даст рабочий retrieval.
3. После этого добавить sparse/hybrid, так как он зависит от уже готовой коллекции и payload.
4. Evaluation делать после появления минимум двух профилей поиска, иначе нечего сравнивать.
5. Reranking можно сначала оставить no-op, но интерфейс лучше заложить до интеграции с генерацией ответа.

## Основные риски

| Риск | Где проявится | Смягчение |
| --- | --- | --- |
| Смешивание embeddings разных моделей | Индексация и query | Manifest, проверка `embedding_model` и `embedding_dim` |
| Нестабильная функция `retrieval_text` | Переиндексация | Unit-тесты и фиксация функции в коде |
| BM25 плохо токенизирует технические обозначения | Sparse retrieval | Evaluation по `GPT-2`, `ETTm1`, `rsLoRA`, числам и дефисам |
| `keywords` неполны | Metadata filters | Не полагаться только на `keywords`; использовать sparse search по `text` |
| Таблицы плохо извлекаются dense-поиском | Запросы по результатам экспериментов | `sparse_text` с полным `text`, профиль `table_numeric`, будущий структурный индекс результатов |
| Рост корпуса | Производительность и стоимость | Batch indexing, cache embeddings, повторная оценка размерности `3072` |

## Definition of Done

ADR считаются реализованными, когда выполняются все условия:

- `chunks/dic_chunks.json` валидируется автоматической командой.
- Qdrant запускается локально и содержит коллекцию `chunks`.
- Коллекция имеет `dense_retrieval` и `sparse_bm25`.
- Все чанки загружены с полным payload, `retrieval_text` и metadata индексации.
- Dense query использует OpenAI `text-embedding-3-large` с размерностью `3072`.
- Sparse query использует BM25-представление `sparse_text`.
- Hybrid query объединяет dense и sparse prefetch через RRF.
- Payload-фильтры работают по ключевым полям ADR-002.
- Reranking-интерфейс принимает полный `text`.
- Evaluation-сценарий сравнивает `dense_only`, `sparse_only`, `hybrid_default`.
- Есть инструкция запуска и переиндексации.

