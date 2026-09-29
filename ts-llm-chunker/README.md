# ts-llm-chunker

Structure-aware chunker для arxiv-статей по теме "LLM for Time Series Forecasting".
Часть пайплайна GraphRAG: берёт MD, полученный из PDF (Marker/Nougat), нормализует его
и сегментирует на чанки по онтологии.

## Архитектура

Два независимых модуля с чёткими контрактами:

- **`src/preprocess/`** — препроцессор. Принимает сырой MD из `articles_raw/`,
  производит нормализованный MD в `articles/`:
  - удаление References / Impact Statement;
  - чистка OCR-артефактов (email, аффилиации, placeholder-ы изображений);
  - разделение склеенных заголовков;
  - сборка разорванных display-формул;
  - вынос метаданных в YAML frontmatter.
- **`src/chunker/`** — чанкер. Принимает нормализованный MD,
  производит `chunks/dic_chunks.json`:
  - иерархический парсинг структуры;
  - маппинг заголовков на онтологические `section` labels;
  - атомарные таблицы и формулы (не разрезаются);
  - recursive split по абзацам с overlap внутри секции.

## Структура репозитория

```
articles_raw/                  # сырой MD (вход препроцессора)
articles/                      # нормализованный MD (вход чанкера)
chunks/                        # выход чанкера (dic_chunks.json)
config/chunker_config.yaml     # гиперпараметры
src/preprocess/                # препроцессор
src/chunker/                   # чанкер
src/common/                    # схемы, io, логи
tests/                         # pytest
doc/                           # проектная документация
```

## Установка

```bash
pip install -e .[dev]
```

## Использование

```bash
# единый запуск: articles_raw/ -> build/articles/ + build/chunks/ + отчёт качества
ts-build --input articles_raw/ --output build/

# препроцессинг: articles_raw/ -> articles/
python -m src.preprocess.run_preprocess \
    --input articles_raw/ \
    --output articles/ \
    --config config/chunker_config.yaml

# чанкинг: articles/ -> chunks/dic_chunks.json
python -m src.chunker.run_chunking \
    --input articles/ \
    --output chunks/ \
    --config config/chunker_config.yaml
```

## Тесты

```bash
pytest
```

## Связь с GraphRAG

Этот модуль автоматически создаёт базовые чанки с `chunk_id`, `article_id`, `section`,
`index`, `text` и `token_count`. Соседний `graph-rag` использует отдельно обогащённый
корпус из 12 статей с метаданными статьи, summary, keywords, типами вопросов и
иерархией секций. Базовый JSON нельзя напрямую индексировать в `graph-rag`: сначала
нужен отдельный этап семантического обогащения и проверка его схемы.

Текущая поддерживаемая стратегия — `recursive`. Крупные таблицы делятся по полным
строкам с повторением заголовка; формулы сохраняются целиком и могут быть крупнее лимита.
