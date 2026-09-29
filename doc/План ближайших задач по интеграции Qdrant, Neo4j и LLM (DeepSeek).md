# План ближайших задач по интеграции Qdrant, Neo4j и LLM (DeepSeek)

**Статус:** In Progress  
**Дата:** 2026-05-22  
**Область:** ближайший этап после стабилизации retrieval- и graph-слоя

## Контекст

К текущему моменту в проекте уже готовы:

- корпус чанков `chunks 2/docs/dic_chunks.json` v1.2.0;
- retrieval-слой в Qdrant;
- графовый слой Neo4j;
- расширение retrieval графовым контекстом;
- сужение доменных связей до chunk-aware evidence.

Следующий рабочий этап — подключить LLM поверх уже готового контура `Qdrant-first + graph context`.

## Архитектурная рамка этапа

На этом этапе принимаются следующие решения:

1. **DeepSeek используется как generation-слой**, а не как замена Qdrant или Neo4j.
2. **Qdrant остаётся первым шагом retrieval**.
3. **Neo4j даёт структурное дополнение**, но не заменяет текстовые чанки.
4. **OpenAI embeddings и DeepSeek generation считаются разными контурами**.
   Если в окружении есть только `DEEPSEEK_API_KEY`, то генерация уже возможна, но query-профили, требующие новых dense embeddings, остаются завязаны на `OPENAI_API_KEY`.
5. Для первого рабочего среза допустим сценарий **`sparse_only + DeepSeek`**, если dense query embeddings недоступны.

## Ближайшие задачи

### Шаг 1. Ввести конфигурацию generation-слоя

- добавить в конфиг секцию `llm`;
- вынести `DEEPSEEK_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL`;
- не смешивать generation-конфиг с dense embedding-конфигом.

### Шаг 2. Собрать сервисный слой ответа

- получить retrieval-результаты из Qdrant;
- при необходимости обогатить их через Neo4j;
- собрать LLM-контекст из:
  - `chunk_id`
  - section path
  - verbatim-текста чанка
  - `question_types`
  - graph summary

### Шаг 3. Добавить CLI-команду верхнего уровня

- реализовать команду `answer`;
- поддержать существующие фильтры retrieval;
- дать возможность запускать сценарий без OpenAI query embeddings через `sparse_only`.

### Шаг 4. Зафиксировать формат grounding

- ответ LLM должен строиться только по переданному контексту;
- вместе с ответом должен печататься список использованных чанков;
- источник истины для Eval API по-прежнему остаётся в самих чанках, а не в свободном ответе модели.

### Шаг 5. Подготовить следующую итерацию

После первого вертикального среза:

- сравнить `sparse_only + DeepSeek` и `hybrid + DeepSeek`;
- оценить, какие relation types реально помогают ответу;
- перейти к валидации на вопросах Eval API.

## Definition of Done

Этап считается минимально завершённым, когда:

- в проекте есть конфиг generation-слоя;
- есть рабочая команда CLI для сценария `retrieve -> expand graph -> ask LLM`;
- ответ сопровождается списком source chunks;
- новый код не ломает `query`, `index-graph`, `check-graph`;
- тесты и локальная smoke-проверка проходят.

## Старт реализации

В рамках текущей итерации стартуем с первых трёх пунктов:

- конфиг `llm`;
- сервис сборки answer-context;
- CLI-команда `answer` под DeepSeek.

## Текущий прогресс

На 2026-05-22 уже выполнено:

- в проект добавлена секция `llm` в конфиг;
- добавлен generation-клиент для OpenAI-compatible API, настроенный под DeepSeek;
- для DeepSeek по умолчанию отключён thinking mode, чтобы не тратить completion budget на `reasoning_content` в базовом GraphRAG-сценарии;
- реализована команда `vector-db-rag answer`;
- `answer` использует текущий retrieval-контур и при необходимости расширяет его графовым контекстом;
- если `OPENAI_API_KEY` отсутствует, answer-сценарий может автоматически перейти в `sparse_only`, чтобы DeepSeek-only окружение не ломалось на этапе query embeddings;
- ответный контекст теперь строится из verbatim-текста чанков, `question_types`, section path и graph summary;
- вместе с ответом печатается список source chunks.
- добавлен `--dry-run`, чтобы собирать grounded context без сетевого вызова LLM;
- добавлен `--json`, чтобы получать структурированный пакет ответа и источников;
- в dry-run режиме можно выводить prompt-данные через `--include-prompts`, что уже пригодно для ручной и автоматической проверки grounding.

Промежуточная проверка:

- `pytest` — зелёный;
- `vector-db-rag --help` и `vector-db-rag answer --help` работают;
- `vector-db-rag answer ... --dry-run` уже отрабатывает локально и показывает grounded context;
- `vector-db-rag answer ... --dry-run --json --include-prompts` уже отдаёт структурированный answer-package;
- подтверждено на реальном DeepSeek API, что при включённом thinking mode малый `max_tokens` может уходить целиком в reasoning; клиент и конфиг после этого скорректированы;
- полный end-to-end вызов `answer` в текущем окружении останавливается только на отсутствии `DEEPSEEK_API_KEY`, что подтверждает готовность локального контура до сетевого шага.
