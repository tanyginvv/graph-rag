# ADR-005: Выбор графовой базы данных для слоя графа знаний GraphRAG-системы

**Статус:** Proposed  
**Дата:** 2026-05-22  
**Автор:** AI-Architect  

## Контекст

В проекте уже зафиксированы решения для слоя поиска:

- ADR-002: векторная база данных — Qdrant;
- ADR-003: dense embedding — `text-embedding-3-large`, `3072`;
- ADR-004: hybrid retrieval — dense + sparse через Qdrant.

На 2026-05-22 завершён семантический чанкинг 12 статей Eval API. Текущий корпус `chunks 2/docs/dic_chunks.json` v1.2.0 содержит:

- **12 статей**;
- **138 чанков**;
- поле `question_types` в каждом чанке;
- `keywords`, привязанные к сущностям и классам онтологии;
- нормализованные поля `article_id`, `chunk_id`, `section`, `section_hierarchy`, `chunk_type`.

Параллельно в `graph-ontology/ontology.md` зафиксирована финальная онтология слоя GraphRAG: **17 классов** и набор доменных связей между ними. Для текущего этапа нужна отдельная графовая БД, которая будет хранить **структурированный граф знаний**, а не dense/sparse поиск.

К графовой БД есть следующие требования:

1. Корректное отображение модели графа свойств из `graph-ontology/ontology.md`: `Article`, `Chunk`, `Method`, `Task`, `Experiment`, `Pipeline`, `Workflow_Step`, `Architecture`, `TuningStrategy`, `Technique`, `DataPreprocessing`, `Dataset`, `Domain`, `Benchmark`, `Metric`, `ExperimentResult`, `TaxonomyCategory`.
2. Поддержка явных связей между чанками и сущностями: как минимум `Article-[:CONTAINS]->Chunk` и `Chunk-[:MENTIONS]->Entity`.
3. Удобные многошаговые обходы графа по типовым сценариям GraphRAG: метод -> архитектура -> техника -> датасет -> метрика -> результат.
4. Простая локальная эксплуатация через Docker на текущей Windows-среде.
5. Прямая интеграция с Python 3.11 и возможность идемпотентной пакетной загрузки.
6. Совместная работа с уже выбранным Qdrant без дублирования логики поиска и без второго конкурирующего векторного слоя.

Важно разделить роли:

- **Qdrant** остаётся слоем поиска и источником dense/sparse candidate retrieval по чанкам.
- **Графовая БД** нужна для слоя графа знаний: явного хранения сущностей, связей, происхождения фактов и расширения контекста через обход графа.

## Решение

**Использовать Neo4j** как графовую базу данных для слоя графа знаний.

Формат внедрения:

- локальная разработка и первичная интеграция — **Neo4j Community** в Docker;
- доступ из Python — **официальный драйвер `neo4j`**;
- язык запросов — **Cypher**;
- Qdrant не заменяется и не дублируется: Neo4j используется как **слой графа знаний и обходов**, а не как основное векторное хранилище.

Первый этап интеграции строится по принципу **двух хранилищ с разной ответственностью**:

| Слой | Хранилище | Ответственность |
| --- | --- | --- |
| Retrieval | Qdrant | dense/sparse индексация, candidate retrieval, payload чанков, фильтры по `article_id`, `question_types`, `keywords` |
| Граф знаний | Neo4j | онтология, сущности, доменные связи, provenance, многошаговый обход |
| Оркестрация | Python-приложение | связывание результатов retrieval и graph expansion через `chunk_id` / `article_id` |

## Детали реализации

### Что именно хранить в Neo4j

Neo4j не должен становиться второй копией Qdrant. Поэтому в графе хранятся:

- **идентификаторы** (`article_id`, `chunk_id`, канонические `id` сущностей);
- **структурные свойства** (`section`, `index`, `chunk_type`, `question_types`);
- **краткие поясняющие поля** (`title`, `name`, `summary`, `year`, `arxiv_id`);
- **доменные связи** и **provenance**.

Полный `text` чанка не является обязательным полем Neo4j на первом этапе. Источником полного текстового фрагмента остаётся Qdrant payload. Это уменьшает дублирование и упрощает синхронизацию.

На первом этапе рекомендуется хранить в `Chunk`:

```json
{
  "chunk_id": "time-llm_chunk_002",
  "article_id": "time-llm",
  "section": "Methodology",
  "index": 2,
  "chunk_type": "prose",
  "question_types": ["А", "Г", "Д", "Е"],
  "token_count": 412,
  "summary": "..."
}
```

### Базовая графовая схема

#### Узлы первого этапа

| Label | Источник | Ключевое свойство |
| --- | --- | --- |
| `Article` | `article_id`, `article_meta.*` из чанков | `article_id` |
| `Chunk` | `chunks 2/docs/dic_chunks.json` | `chunk_id` |
| `Method` | нормализованные `keywords` / извлечение сущностей | `id` |
| `Architecture` | нормализованные `keywords` / извлечение сущностей | `id` |
| `Technique` | нормализованные `keywords` / извлечение сущностей | `id` |
| `Dataset` | нормализованные `keywords` / извлечение сущностей | `id` |
| `Benchmark` | нормализованные `keywords` / извлечение сущностей | `id` |
| `Metric` | нормализованные `keywords` / извлечение сущностей | `id` |
| `Domain` | нормализованные `keywords` / извлечение сущностей | `id` |
| `Task` | нормализованные `keywords` / извлечение сущностей | `id` |
| `TaxonomyCategory` | только реальные категории таксономии из статей, не буквы `А–М` | `id` |

#### Связи первого этапа

| Связь | Назначение |
| --- | --- |
| `(:Article)-[:CONTAINS]->(:Chunk)` | источник чанка |
| `(:Chunk)-[:MENTIONS]->(:Method)` | связь чанка с методом |
| `(:Chunk)-[:MENTIONS]->(:Architecture)` | связь чанка с архитектурой |
| `(:Chunk)-[:MENTIONS]->(:Technique)` | связь чанка с техникой |
| `(:Chunk)-[:MENTIONS]->(:Dataset)` | связь чанка с датасетом |
| `(:Chunk)-[:MENTIONS]->(:Benchmark)` | связь чанка с бенчмарком |
| `(:Chunk)-[:MENTIONS]->(:Metric)` | связь чанка с метрикой |
| `(:Chunk)-[:MENTIONS]->(:Domain)` | связь чанка с доменом |
| `(:Chunk)-[:MENTIONS]->(:Task)` | связь чанка с задачей |
| `(:Chunk)-[:MENTIONS]->(:TaxonomyCategory)` | связь чанка с категорией таксономии |

### Важное разграничение: `question_types` и `TaxonomyCategory`

Поле `question_types` из корпуса чанков **не должно автоматически превращаться** в узлы `TaxonomyCategory`.

Причина:

- `question_types` — это таксономия вопросов Eval API (`А`, `А+`, `В`, ..., `Спец`);
- `TaxonomyCategory` в онтологии — это доменные категории методов/подходов, а не служебные типы оценочных вопросов.

Поэтому на первом этапе:

- `question_types` хранится как **свойство узла `Chunk`**;
- `TaxonomyCategory` создаётся только там, где в статьях действительно выделена предметная категория;
- логика "какие классы онтологии вероятнее всего упоминаются в чанке данного типа вопроса" реализуется в приложении, а не через отдельные узлы `QuestionType`.

### Второй этап: обогащение доменных связей

После первого этапа граф расширяется до доменной схемы `ontology.md`:

| Связь | Когда добавляется |
| --- | --- |
| `(:Article)-[:PROPOSES]->(:Method)` | после нормализации главного метода статьи |
| `(:Article)-[:EVALUATES]->(:Method)` | после извлечения baselines |
| `(:Article)-[:SURVEYS]->(:TaxonomyCategory)` | для обзорных статей |
| `(:Article)-[:ADDRESSES]->(:Task)` | после извлечения задач |
| `(:Article)-[:CONDUCTS]->(:Experiment)` | после выделения экспериментов |
| `(:Method)-[:HAS_ARCHITECTURE]->(:Architecture)` | после нормализации backbone |
| `(:Method)-[:HAS_TUNING]->(:TuningStrategy)` | после извлечения стратегии дообучения |
| `(:Method)-[:USES_TECHNIQUE]->(:Technique)` | после детализации техники |
| `(:Method)-[:USES_PREPROCESSING]->(:DataPreprocessing)` | после выделения предобработки |
| `(:Method)-[:HAS_PIPELINE]->(:Pipeline)` | после формализации пайплайна |
| `(:Pipeline)-[:CONTAINS]->(:Workflow_Step)` | после разметки шагов |
| `(:Experiment)-[:FOR_TASK]->(:Task)` | после структурирования экспериментов |
| `(:Experiment)-[:ON_DATASET]->(:Dataset)` | после структурирования экспериментов |
| `(:Experiment)-[:USES_BENCHMARK]->(:Benchmark)` | после структурирования экспериментов |
| `(:Experiment)-[:HAS_RESULT]->(:ExperimentResult)` | после извлечения численных результатов |
| `(:ExperimentResult)-[:MEASURED_BY]->(:Metric)` | после извлечения метрики |
| `(:ExperimentResult)-[:OF_METHOD]->(:Method)` | после связывания результата с методом |
| `(:Dataset)-[:BELONGS_TO]->(:Domain)` | после нормализации домена |
| `(:Method)-[:BELONGS_TO]->(:TaxonomyCategory)` | для категориальных обзоров и survey-сценариев |

Это расширение не требуется для первого технического запуска GraphRAG. Минимально полезный граф — это `Article`, `Chunk`, core entities и `MENTIONS`.

### Канонические идентификаторы

Для всех сущностей, кроме `Article` и `Chunk`, вводится отдельное поле `id`.

Рекомендуемый принцип:

- `Article.article_id = article_id` из Eval API;
- `Chunk.chunk_id = chunk_id` из корпуса;
- для остальных сущностей: `id = "<Label>:<normalized_name>"`.

Примеры:

- `Method:time-llm`
- `Architecture:gpt-2`
- `Dataset:ettm1`
- `Metric:mse`

При этом:

- `name` хранит каноническое человекочитаемое значение;
- `aliases` можно добавлять позже для синонимов и написаний (`GPT2`, `GPT-2`, `OpenAI GPT-2`);
- `source_chunks` как массив лучше не хранить в узлах: provenance должно жить либо на ребре `MENTIONS`, либо вычисляться запросом.

### Provenance

Каждая связь, извлечённая из чанка, должна быть трассируема обратно к источнику.

Минимальный набор provenance-свойств на отношениях `MENTIONS` и производных доменных связях:

```json
{
  "source_chunk_id": "time-llm_chunk_002",
  "article_id": "time-llm",
  "extraction_source": "keywords",
  "question_types": ["А", "Г", "Д", "Е"]
}
```

Если позже связь строится LLM-экстракцией, меняется `extraction_source`, например на `llm_extraction`, и добавляется `confidence`.

### Ограничения и индексы

В Neo4j Community для прототипа достаточно **уникальных ограничений** на стабильные ключи. Обязательность полей контролируется приложением и Pydantic-моделями, так как часть property existence constraints доступна только в Enterprise-редакции.

Минимальные ограничения:

```cypher
CREATE CONSTRAINT article_id_unique IF NOT EXISTS
FOR (a:Article) REQUIRE a.article_id IS UNIQUE;

CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE;

CREATE CONSTRAINT method_id_unique IF NOT EXISTS
FOR (n:Method) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT architecture_id_unique IF NOT EXISTS
FOR (n:Architecture) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT technique_id_unique IF NOT EXISTS
FOR (n:Technique) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT dataset_id_unique IF NOT EXISTS
FOR (n:Dataset) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT benchmark_id_unique IF NOT EXISTS
FOR (n:Benchmark) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT metric_id_unique IF NOT EXISTS
FOR (n:Metric) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT domain_id_unique IF NOT EXISTS
FOR (n:Domain) REQUIRE n.id IS UNIQUE;

CREATE CONSTRAINT task_id_unique IF NOT EXISTS
FOR (n:Task) REQUIRE n.id IS UNIQUE;
```

Для `TaxonomyCategory`, `Pipeline`, `Workflow_Step`, `TuningStrategy`, `DataPreprocessing`, `Experiment`, `ExperimentResult` ограничения добавляются по мере появления этих узлов в рабочем пайплайне.

### Стратегия загрузки

Загрузка в Neo4j должна быть **идемпотентной** и выполняться пакетами через `UNWIND` + `MERGE`.

Порядок:

1. Прочитать `chunks 2/docs/dic_chunks.json`.
2. Сгруппировать чанки по `article_id`.
3. Создать/обновить `Article`.
4. Создать/обновить `Chunk`.
5. Создать `(:Article)-[:CONTAINS]->(:Chunk)`.
6. Нормализовать `keywords` в сущности нужных классов.
7. Создать/обновить узлы сущностей.
8. Создать `MENTIONS`-связи с provenance.

Рекомендуемый шаблон Cypher:

```cypher
UNWIND $rows AS row
MERGE (a:Article {article_id: row.article_id})
SET a.title = row.article_title,
    a.year = row.year,
    a.arxiv_id = row.arxiv_id

MERGE (c:Chunk {chunk_id: row.chunk_id})
SET c.article_id = row.article_id,
    c.section = row.section,
    c.index = row.index,
    c.chunk_type = row.chunk_type,
    c.question_types = row.question_types,
    c.token_count = row.token_count,
    c.summary = row.summary

MERGE (a)-[:CONTAINS]->(c)
```

Отдельный пакет для `MENTIONS`:

```cypher
UNWIND $rows AS row
MATCH (c:Chunk {chunk_id: row.chunk_id})
MATCH (e {id: row.entity_id})
MERGE (c)-[r:MENTIONS]->(e)
SET r.article_id = row.article_id,
    r.source_chunk_id = row.chunk_id,
    r.extraction_source = row.extraction_source,
    r.question_types = row.question_types
```

Замечание по `MERGE`: безопаснее делать отдельные `MERGE` по узлам и отдельный `MERGE` по связи, а не слеплять большой шаблон в один `MERGE`.

### Интеграция с Qdrant

Базовый рабочий сценарий:

1. Пользовательский запрос идёт в Qdrant.
2. Qdrant возвращает top-K чанков по dense/sparse/hybrid retrieval.
3. Приложение извлекает их `chunk_id`.
4. По `chunk_id` выполняется Cypher-запрос в Neo4j.
5. Из графа достаются:
   - сущности, которые упомянуты в найденных чанках;
   - ближайшие доменные связи вокруг этих сущностей;
   - агрегированный структурный контекст для ответа и объяснения.

Пример graph expansion:

```cypher
MATCH (c:Chunk)
WHERE c.chunk_id IN $chunk_ids
OPTIONAL MATCH (c)-[:MENTIONS]->(e)
OPTIONAL MATCH (e)-[r*1..2]-(n)
RETURN c, e, r, n
```

Практический смысл:

- Qdrant отвечает на вопрос: **какие чанки текстово релевантны?**
- Neo4j отвечает на вопрос: **как связаны сущности внутри этих чанков и вокруг них?**

На первом этапе основным остаётся сценарий **Qdrant-first**. Сценарии graph-first через чистый Cypher допустимы позже, но не являются основной стратегией для Eval API.

### Развёртывание

Рекомендуемый режим локального старта:

- Docker-контейнер Neo4j Community;
- публикация портов `7474` (Browser) и `7687` (Bolt);
- обязательный volume для данных;
- явная фиксация версии образа в `docker-compose.yml`.

Иллюстративно:

```yaml
services:
  neo4j:
    image: neo4j:<pinned-version>
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      - NEO4J_AUTH=neo4j/change-me
    volumes:
      - neo4j_data:/data

volumes:
  neo4j_data:
```

## Причины выбора Neo4j

1. **Прямое соответствие модели графа свойств проекта.** Онтология `graph-ontology/ontology.md` естественно выражается через размеченные узлы, типизированные связи и Cypher-паттерны.
2. **Cypher хорошо подходит для GraphRAG traversal.** Для будущих сценариев важны path-patterns, multi-hop traversal и shortest path-поиск по структурированному графу.
3. **Официальный Python-драйвер и управляемые транзакции.** Это упрощает пакетную загрузку, идемпотентные `MERGE`-операции и батчевую обработку из существующего Python-пайплайна.
4. **Простой локальный запуск через Docker.** Для проекта в активной разработке это снижает порог эксплуатации и упрощает воспроизводимость.
5. **Совместимость с уже выбранным Qdrant.** В официальной документации `neo4j-graphrag-python` поддерживаются external retrievers, включая Qdrant, то есть экосистемный конфликт между выбранной векторной БД и Neo4j отсутствует.
6. **Community Edition достаточно для прототипа.** Для первого этапа нам нужны уникальные ограничения, Cypher и транзакции; Enterprise-функции не обязательны.

## Рассмотренные альтернативы

| Альтернатива | Причина отклонения |
| --- | --- |
| **Memgraph** | Сильный кандидат: Cypher, Python-экосистема, явная ориентация на GraphRAG. Однако официальные материалы Memgraph описывают систему как in-memory graph database и продвигают отдельный стек `GQLAlchemy`. Для текущего проекта, где граф должен быть максимально консервативным и предсказуемым дополнением к Qdrant, Neo4j выглядит более безопасным выбором по зрелости экосистемы и количеству стандартных практик. Это инженерный вывод, а не утверждение о непригодности Memgraph. |
| **ArangoDB** | Официально поддерживает named graphs, traversal в AQL и официальный Python-драйвер. Но проекту не нужна multi-model БД: retrieval уже вынесен в Qdrant, а онтология выражается обычным property graph. Переход на AQL добавляет вторую концептуальную модель без явного выигрыша. |
| **Neo4j как единственное хранилище, включая vector search** | Neo4j поддерживает vector indexes, но проект уже стандартизирован вокруг Qdrant по ADR-002/003/004. Отказ от Qdrant сейчас означал бы пересмотр уже реализованного retrieval-слоя и риск синхронной миграции двух подсистем сразу. |
| **PostgreSQL + graph-расширение** | Возможен, если бы проект уже жил вокруг Postgres. Но сейчас требуется именно нативный графовый слой с Cypher-подобными traversal-запросами и минимальной дополнительной инфраструктурой. |

## Последствия

### Положительные

- Получаем явный слой графа знаний, не смешанный с векторным поиском.
- Появляется удобная модель provenance: любой факт можно связать с `chunk_id`.
- GraphRAG становится объяснимее: кроме найденных чанков можно показать путь по сущностям и отношениям.
- Разделение ролей между Qdrant и Neo4j делает архитектуру проще: текстовый поиск отдельно, структурный граф отдельно.
- Открывается путь к direct Cypher-сценариям для вопросов вида "какие методы используют GPT-2 и сравниваются на ETTm1?".

### Риски и требования

- Появляется **двуххранилищная синхронизация**. Ключи `article_id` и `chunk_id` должны быть строго одинаковыми в Qdrant и Neo4j.
- Качество графа будет зависеть от качества нормализации `keywords` и последующего entity extraction.
- Neo4j Community не закроет все ограничения схемы на уровне БД; часть проверок останется на приложении.
- Если позднее потребуется хранить в графе большие текстовые свойства или строить graph-only UI без обращения к Qdrant, модель хранения придётся расширить.
- Нельзя смешивать `question_types` Eval API и `TaxonomyCategory` онтологии: это разные уровни семантики.

## Что делать дальше

1. Добавить в репозиторий локальный `docker-compose` для Neo4j.
2. Создать модуль `src/vector_db_rag/graph_store.py` или отдельный пакет для Neo4j-клиента.
3. Реализовать минимальный импорт:
   - `Article`
   - `Chunk`
   - `CONTAINS`
   - core entity nodes
   - `MENTIONS`
4. Зафиксировать правила нормализации `keywords -> ontology class`.
5. После первого импорта добавить один диагностический CLI-командный путь:
   - `index-graph`
   - `check-graph`
   - `expand-chunks --chunk-id ...`

## Связанные решения

- ADR-001 — стратегия чанкинга и структура обогащённого чанка.
- ADR-002 — выбор Qdrant как векторной базы данных.
- ADR-003 — выбор dense embedding-модели и размерности.
- ADR-004 — стратегия sparse/hybrid search.
- `graph-ontology/ontology.md` — целевая онтология графа знаний.
- `chunks/doc/Структура и содержимое чанков.md` — схема чанка v1.2.

## Источники

### Локальные документы проекта

- `graph-ontology/ontology.md`
- `chunks 2/docs/dic_chunks.json`
- `chunks/doc/Структура и содержимое чанков.md`
- `doc/ADR-Выбор векторной базы данных для GraphRAG-системы.md`
- `doc/ADR-Выбор embedding-модели и размерности для GraphRAG-системы.md`
- `doc/ADR-Стратегия sparse и hybrid search для GraphRAG-системы.md`

### Внешние официальные источники

- Neo4j Operations Manual: Docker deployment  
  https://neo4j.com/docs/operations-manual/current/docker/introduction/
- Neo4j Python Driver Manual: transactions  
  https://neo4j.com/docs/python-manual/current/transactions/
- Neo4j Cypher Manual: constraints  
  https://neo4j.com/docs/cypher-manual/current/schema/constraints/
- Neo4j Cypher Manual: shortest paths and path patterns  
  https://neo4j.com/docs/cypher-manual/current/patterns/shortest-paths/
- Neo4j GraphRAG for Python: external retrievers, включая Qdrant  
  https://neo4j.com/docs/neo4j-graphrag-python/current/index.html
- Memgraph: product overview  
  https://memgraph.com/
- Memgraph for Python developers / GQLAlchemy  
  https://memgraph.com/memgraph-for-python-developers
- ArangoDB Documentation: graphs in AQL  
  https://docs.arangodb.com/3.12/aql/graphs/
- ArangoDB Documentation: Python driver  
  https://docs.arangodb.com/3.11/develop/drivers/python/
