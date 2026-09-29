# ts-llm-chunker: Подробная документация

Structure-aware chunker для научных статей arxiv по теме **"LLM for Time Series Forecasting"**.
Модуль является частью пайплайна GraphRAG: принимает Markdown, полученный из PDF (Marker/Nougat),
нормализует его и сегментирует на чанки, размеченные по онтологии предметной области.

---

## Оглавление

1. [Общая архитектура](#1-общая-архитектура)
2. [Структура репозитория](#2-структура-репозитория)
3. [Пайплайн обработки](#3-пайплайн-обработки)
   - [3.1. Препроцессинг](#31-препроцессинг-articles_raw--articles)
   - [3.2. Чанкинг](#32-чанкинг-articles--chunks)
4. [Описание модулей и классов](#4-описание-модулей-и-классов)
   - [4.1. src/common/](#41-srccommon--общие-компоненты)
   - [4.2. src/preprocess/](#42-srcpreprocess--препроцессор)
   - [4.3. src/chunker/](#43-srcchunker--чанкер)
5. [Конфигурация](#5-конфигурация)
6. [Примеры использования](#6-примеры-использования)
7. [Тестирование](#7-тестирование)

---

## 1. Общая архитектура

Модуль состоит из двух независимых подсистем с чётким контрактом:

```
                    articles_raw/*.md                    articles/*.md                     chunks/
 PDF -> Marker ->  [сырой Markdown]  -> Препроцессор -> [чистый Markdown] -> Чанкер ->  [dic_chunks.json]
                                                                                         [per_article/*.json]
```

**Препроцессор** (`src/preprocess/`) очищает сырой MD от OCR-артефактов, нормализует структуру
и формирует YAML frontmatter с метаданными статьи.

**Чанкер** (`src/chunker/`) парсит структуру нормализованного MD, группирует блоки по секциям
согласно онтологии и выполняет recursive split. Крупные таблицы делятся по целым строкам с повторением заголовка; формулы сохраняются целиком.

Принцип разделения: **препроцессор готовит чистый MD как самостоятельный артефакт**
(пригодный для ручной инспекции, хранения в git, использования другими downstream-потребителями),
**чанкер принимает чистый MD и производит сегментацию**.

---

## 2. Структура репозитория

```
ts-llm-chunker/
├── pyproject.toml                          # зависимости, CLI entry points, настройки инструментов
├── .gitignore
├── README.md                               # краткое описание проекта
│
├── config/
│   └── chunker_config.yaml                 # гиперпараметры препроцессора и чанкера
│
├── doc/
│   ├── README.md                           # данный документ (подробная документация)
│   ├── План реализации репозитория...md     # план реализации модуля
│   └── Разделение зон ответственности...md # разграничение конвертер/чанкер
│
├── articles_raw/                           # сырой MD из PDF-конвертера (вход препроцессора)
│   ├── 2403_05798.md                       # arXiv-формат (##/### иерархия)
│   └── 2603.29756.md                       # IEEE-формат  (#### плоская иерархия)
│
├── articles/                               # нормализованный MD (выход препроцессора, вход чанкера)
│   ├── 2403_05798.md
│   └── 2603.29756.md
│
├── chunks/                                 # выход чанкера
│   ├── dic_chunks.json                     # все чанки всех статей (ChunksDocument)
│   └── per_article/
│       ├── 2403_05798.json                 # чанки одной статьи (list[Chunk])
│       └── 2603.29756.json
│
├── src/
│   ├── common/                             # общие компоненты
│   │   ├── schemas.py                      # Pydantic-модели: ArticleMetadata, Chunk, ChunksDocument
│   │   ├── io_utils.py                     # чтение/запись файлов, YAML-конфиг, хэширование
│   │   └── logging_config.py               # единая настройка логирования
│   │
│   ├── preprocess/                         # модуль препроцессинга
│   │   ├── run_preprocess.py               # CLI entry point (ts-preprocess)
│   │   ├── md_cleaner.py                   # оркестратор: последовательный вызов всех чисток
│   │   ├── header_normalizer.py            # разделение склеенных заголовков, удаление строки авторов
│   │   ├── artifact_cleaner.py             # удаление OCR-артефактов: image placeholders, email, аффилиации
│   │   ├── math_normalizer.py              # сборка разорванных display-формул, склейка параграфов
│   │   ├── section_remover.py              # удаление секций References и Impact Statement
│   │   └── frontmatter_builder.py          # сборка/нормализация YAML frontmatter
│   │
│   └── chunker/                            # модуль чанкинга
│       ├── run_chunking.py                 # CLI entry point (ts-chunk)
│       ├── chunker.py                      # оркестратор: parse -> group -> mask -> split -> unmask
│       ├── structure_parser.py             # парсинг MD в типизированные блоки (MDBlock)
│       ├── section_mapper.py               # маппинг заголовков на онтологические labels (SectionLabel)
│       ├── math_masker.py                  # маскирование/восстановление LaTeX-формул
│       ├── token_counter.py                # подсчёт токенов через tiktoken
│       └── splitters/
│           └── recursive_splitter.py       # recursive split: подзаголовки -> абзацы -> предложения
│
└── tests/
    ├── preprocess/
    │   ├── test_artifact_cleaner.py
    │   ├── test_section_remover.py
    │   ├── test_header_normalizer.py
    │   ├── test_math_normalizer.py
    │   └── test_frontmatter_builder.py
    └── chunker/
        ├── test_structure_parser.py
        ├── test_math_masker.py
        ├── test_recursive_splitter.py
        ├── test_section_mapper.py
        └── test_chunker.py
```

---

## 3. Пайплайн обработки

### 3.1. Препроцессинг (`articles_raw/` -> `articles/`)

Оркестратор: `md_cleaner.clean_md()`. Этапы выполняются последовательно:

```
raw_md
  -> header_normalizer.normalize_headers()     # 1. Заголовки
  -> artifact_cleaner.clean_artifacts()         # 2. OCR-артефакты
  -> math_normalizer.normalize_math()           # 3. Формулы и параграфы
  -> section_remover.remove_sections()          # 4. Удаление секций
  -> frontmatter_builder.ensure_frontmatter()   # 5. YAML frontmatter
  -> trailing whitespace cleanup                # 6. Финальная чистка
  = clean_md
```

| Этап | Модуль | Что делает |
|------|--------|------------|
| 1 | `header_normalizer` | Разделяет склеенные заголовки (`текст # Heading` -> две строки). Удаляет строку авторов после `# Title` (авторы уже в frontmatter) |
| 2 | `artifact_cleaner` | Удаляет: image placeholders `![](_page_N_...)`, email-адреса `<email@...>`, хвосты аффилиаций с `Correspondence to:`, пустые строки таблиц |
| 3 | `math_normalizer` | Склеивает разорванные display-формулы (`$$expr1$$` + `$$= expr2$$` -> одна формула). Склеивает ложные разрывы параграфов (предложение разорвано посередине) |
| 4 | `section_remover` | Удаляет секции References и Impact Statement целиком (от заголовка до следующего заголовка того же или более высокого уровня) |
| 5 | `frontmatter_builder` | Обеспечивает наличие YAML frontmatter с полями: `article_id`, `title`, `authors`, `year`, `arxiv_id`, `venue`. Дополняет из имени файла, если поля пусты |

### 3.2. Чанкинг (`articles/` -> `chunks/`)

Оркестратор: `chunker.chunk_article()`. Этапы:

```
clean_md
  -> frontmatter -> ArticleMetadata              # 1. Извлечение метаданных
  -> structure_parser.parse_blocks()              # 2. Парсинг блочной структуры
  -> chunker._group_by_section()                  # 3. Группировка по секциям
  -> for each section:
       math_masker.mask_formulas()                # 4a. Маскирование формул
       recursive_splitter.recursive_split()       # 4b. Recursive split
       math_masker.unmask_formulas()              # 4c. Восстановление формул
  -> list[Chunk]                                  # 5. Формирование чанков
```

| Этап | Модуль | Что делает |
|------|--------|------------|
| 1 | `chunker._extract_metadata` | Извлекает `ArticleMetadata` из YAML frontmatter |
| 2 | `structure_parser` | Парсит MD в список `MDBlock` с типами: `heading`, `paragraph`, `table`, `display_math`, `figure_caption` |
| 3 | `chunker._group_by_section` | Адаптивно группирует блоки по секциям (см. [ниже](#адаптивная-группировка-по-секциям)) |
| 4a | `math_masker` | Заменяет `$$...$$` и `$...$` на плейсхолдеры `[DISPLAY_MATH_NNN]` / `[MATH_NNN]` |
| 4b | `recursive_splitter` | Разбивает блоки секции на чанки: подзаголовки -> абзацы -> предложения; крупные таблицы делит по строкам, display math не разрезает |
| 4c | `math_masker` | Восстанавливает формулы из плейсхолдеров |
| 5 | `chunker` | Оборачивает каждый текстовый фрагмент в `Chunk` с `chunk_id`, `section`, `index`, `token_count` |

#### Адаптивная группировка по секциям

Модуль адаптируется к разным форматам научных статей:

**arXiv-формат** (многоуровневая иерархия):
```markdown
## 1. Introduction        <- top_level = 2, секция
### 1.1 Background         <- подзаголовок (level 3 > top_level) -> часть секции
## 2. Related Work         <- секция
```
Стратегия: любой заголовок на уровне `<= top_level` создаёт новую секцию.

**IEEE-формат** (плоская иерархия):
```markdown
#### I. INTRODUCTION       <- top_level = 4, секция (распознаётся по паттерну)
#### A. Background          <- тот же уровень, но подсекция -> НЕ boundary
#### II. RELATED WORK       <- секция (распознаётся по паттерну)
## Supplementary            <- level < top_level -> всегда boundary
```
Стратегия: на уровне `top_level` граница секции определяется по **pattern-matching** заголовка
(только main-body labels: Introduction, RelatedWork, Methodology и т.д.).
Заголовки уровнем выше (меньше числом) `top_level` всегда создают boundary.

Алгоритм:
1. `_detect_top_heading_level()` — определяет `top_level` по первому заголовку с `level >= 2`
2. `_has_sub_headings()` — проверяет наличие подзаголовков глубже `top_level`
3. `_is_section_boundary()` — решает, является ли заголовок границей секции

---

## 4. Описание модулей и классов

### 4.1. `src/common/` — общие компоненты

#### `schemas.py`

Pydantic-модели, определяющие контракт между модулями:

- **`SectionLabel`** — Literal-тип из 16 значений онтологических меток секций:
  `Abstract`, `Introduction`, `RelatedWork`, `Preliminaries`, `Methodology`,
  `Experiments`, `Results`, `Discussion`, `Conclusion`,
  `Appendix_Implementation`, `Appendix_Baselines`, `Appendix_Datasets`,
  `Appendix_Metrics`, `Appendix_Results`, `Appendix_Other`, `Unknown`.

- **`ArticleMetadata`** — метаданные статьи из YAML frontmatter:
  - `article_id: str` — стабильный идентификатор (напр. `arxiv_2403_05798`)
  - `title: str`
  - `authors: list[str]`
  - `year: int | None`
  - `arxiv_id: str | None` — идентификатор arXiv (напр. `2403.05798`)
  - `venue: str | None`

- **`Chunk`** — финальный чанк для GraphRAG:
  - `chunk_id: str` — уникальный id (напр. `arxiv_2403_05798_chunk_007`)
  - `article_id: str`
  - `section: SectionLabel` — онтологическая метка секции
  - `index: int` — порядковый номер в статье (с нуля)
  - `text: str` — текст чанка
  - `token_count: int` — количество токенов (tiktoken, cl100k_base)

- **`ChunksDocument`** — обёртка для `dic_chunks.json`:
  - `chunker_version: str`
  - `config_hash: str` — SHA1-хэш конфигурации для версионирования
  - `chunks: list[Chunk]`

#### `io_utils.py`

Утилиты ввода/вывода:
- `read_text()` / `write_text()` — чтение/запись текстовых файлов (UTF-8)
- `read_json()` / `write_json()` — чтение/запись JSON с автосозданием директорий
- `load_config()` — загрузка YAML-конфигурации
- `config_hash()` — стабильный SHA1-хэш конфига
- `iter_md_files()` — список `.md` файлов в директории (отсортированный)

#### `logging_config.py`

- `setup_logging(level)` — настройка root logger (идемпотентно)
- `get_logger(name)` — получение именованного логгера

### 4.2. `src/preprocess/` — препроцессор

#### `md_cleaner.py` — оркестратор

Функция `clean_md()` принимает сырой MD-текст и последовательно применяет все этапы очистки.
Параметры позволяют отключить отдельные чистки (`remove_references`, `remove_images` и т.д.).

#### `header_normalizer.py`

- `split_glued_headings(text)` — разделяет строки вида `текст # Heading`
  на две отдельные строки (OCR-артефакт склеивания).
- `remove_author_line(text)` — удаляет строку перечня авторов после `# Title`
  (авторы уже вынесены в YAML frontmatter).
- `normalize_headers(text)` — применяет оба преобразования.

#### `artifact_cleaner.py`

- `remove_image_placeholders(text)` — удаляет строки `![](_page_N_Figure_M.jpeg)`
- `remove_empty_table_rows(text)` — удаляет строки таблиц из пустых ячеек `| | | |`
- `remove_affiliation_tails(text)` — удаляет хвосты аффилиаций с `Correspondence to:`
- `remove_inline_emails(text)` — удаляет email в угловых скобках `<email@example.com>`
- `clean_artifacts(text, ...)` — применяет все чистки последовательно

#### `math_normalizer.py`

- `merge_broken_display_math(text)` — склеивает разорванные display-формулы:
  - `$$expr1$$` + `$$= expr2$$` -> `$$expr1 = expr2$$`
  - `$$expr$$` + `(3)` -> `$$expr \tag{3}$$`
- `join_broken_paragraphs(text)` — склеивает ложные разрывы параграфов
  (предложение не завершено пунктуацией, следующая строка начинается с маленькой буквы).
- `normalize_math(text)` — применяет оба преобразования.

#### `section_remover.py`

- `remove_sections(text, ...)` — удаляет секции References и/или Impact Statement
  целиком: от заголовка до следующего заголовка того же или более высокого уровня.

#### `frontmatter_builder.py`

- `article_id_from_filename(filename)` — формирует `article_id` из имени файла:
  `2403_05798.md` -> `arxiv_2403_05798`, `2603.29756.md` -> `arxiv_2603_29756`
- `arxiv_id_from_filename(filename)` — извлекает arXiv ID: `2403_05798.md` -> `2403.05798`
- `ensure_frontmatter(text, filename)` — гарантирует наличие YAML frontmatter с полями
  `article_id`, `arxiv_id`, `title`, `authors`, `year`, `venue`.

### 4.3. `src/chunker/` — чанкер

#### `structure_parser.py`

Датакласс **`MDBlock`** — один логический блок MD-документа:
- `type: BlockType` — `"heading"` | `"paragraph"` | `"table"` | `"display_math"` | `"figure_caption"`
- `content: str` — текстовое содержимое блока
- `heading_level: int | None` — уровень заголовка (1-6)
- `heading_number: str | None` — номер заголовка (`"3.2"`, `"A.1"`)
- `start_line: int`, `end_line: int` — позиция в исходном файле

Функция `parse_blocks(text)` парсит MD-текст в `list[MDBlock]`:
- Автоматически пропускает YAML frontmatter
- Распознаёт заголовки (`# ...`), таблицы (`|...|`), display math (`$$...$$`),
  figure/table captions (`Figure N...`, `Table N...`), остальное — paragraph

#### `section_mapper.py`

Функция `classify_heading(heading_content)` определяет `SectionLabel` для заголовка
по regex-паттернам. Поддерживает:
- Арабские номера секций: `1.`, `3.1.`, `5.`
- Римские номера (IEEE-формат): `I.`, `II.`, `III.`, `IV.`, `V.`, `VI.`, `VII.`, `VIII.`
- Буквенные номера приложений: `A.`, `B.1.`, `C.`
- Заголовки с `#`-префиксом и без него

Константа `SECTION_PATTERNS` содержит упорядоченный список пар `(regex, SectionLabel)`.
Первое совпадение побеждает.

#### `math_masker.py`

- `mask_formulas(text)` -> `(masked_text, mapping)` — заменяет LaTeX-формулы на плейсхолдеры:
  - Display math `$$...$$` -> `[DISPLAY_MATH_NNN]`
  - Inline math `$...$` -> `[MATH_NNN]`
  - Порядок важен: сначала display, потом inline
- `unmask_formulas(text, mapping)` — восстанавливает формулы из плейсхолдеров

Маскирование необходимо, чтобы recursive split не разрезал формулы посередине
при подсчёте токенов и поиске точек разбиения.

#### `token_counter.py`

- `count_tokens(text, encoding_name)` — подсчёт токенов через `tiktoken`
  (по умолчанию `cl100k_base`, соответствует GPT-4)

#### `chunker.py` — оркестратор

Ключевые внутренние функции:

- `_extract_metadata(text, filename)` — извлечение `ArticleMetadata` из frontmatter
- `_detect_top_heading_level(blocks)` — определение верхнего уровня заголовков основного текста
  (эвристика: первый heading с `level >= 2`)
- `_has_sub_headings(blocks, top_level)` — проверка наличия подзаголовков глубже `top_level`
- `_is_section_boundary(block, top_level, flat_hierarchy)` — решение, является ли заголовок
  границей секции (с учётом формата статьи)
- `_group_by_section(blocks)` — адаптивная группировка блоков по секциям

Публичная функция:

- `chunk_article(text, *, filename, max_chunk_size, min_chunk_size, overlap_tokens, encoding)`
  -> `(ArticleMetadata, list[Chunk])` — полный пайплайн чанкинга одной статьи

#### `splitters/recursive_splitter.py`

Функция `recursive_split(blocks, ...)` разбивает блоки одной секции на текстовые чанки:

1. **Структурные блоки**: небольшая таблица или `display_math` образует отдельный чанк; крупная таблица делится по строкам с повторением заголовка
2. **Текстовые блоки** объединяются и разбиваются рекурсивно:
   - По абзацам (двойной перенос строки)
   - Если абзац > `max_chunk_size` — по предложениям
3. Мелкие куски объединяются до `max_chunk_size`
4. Overlap (перекрытие) добавляется между чанками внутри секции

Вспомогательные функции:
- `_merge_small_chunks()` — объединение мелких кусков
- `_add_overlap()` — добавление overlap (последние предложения предыдущего чанка)

#### `run_chunking.py` — CLI entry point

Обрабатывает все `.md` файлы из `articles/`, пишет:
- `chunks/per_article/{stem}.json` — чанки каждой статьи отдельно
- `chunks/dic_chunks.json` — все чанки в единой обёртке `ChunksDocument`

---

## 5. Конфигурация

Файл `config/chunker_config.yaml`:

```yaml
chunker:
  max_chunk_size: 600        # предел для текста; формулы и отдельные длинные строки таблиц могут превышать
  min_chunk_size: 150        # целевой нижний предел (не жёсткий)
  overlap_tokens: 80         # количество токенов перекрытия между чанками
  strategy: "recursive"
  tokenizer: "cl100k_base"   # tiktoken encoding (GPT-4)

semantic_splitter:           # (опционально, для стратегии "semantic")
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  breakpoint_percentile: 95
  buffer_size: 1

preprocess:
  remove_references: true
  remove_impact_statement: true
  remove_image_placeholders: true
  remove_emails: true
```

---

## 6. Примеры использования

### Установка

```bash
pip install -e .[dev]
```

### Запуск полного пайплайна

```bash
# Шаг 1: Препроцессинг
python -m src.preprocess.run_preprocess \
    --input articles_raw/ \
    --output articles/ \
    --config config/chunker_config.yaml

# Шаг 2: Чанкинг
python -m src.chunker.run_chunking \
    --input articles/ \
    --output chunks/ \
    --config config/chunker_config.yaml
```

Или через зарегистрированные CLI entry points:

```bash
ts-preprocess --input articles_raw/ --output articles/
ts-chunk --input articles/ --output chunks/
```

### Использование из Python-кода

#### Препроцессинг одной статьи

```python
from src.preprocess.md_cleaner import clean_md

raw_md = open("articles_raw/2403_05798.md").read()
clean = clean_md(raw_md, filename="2403_05798.md")
print(clean[:500])
```

Результат:
```yaml
---
article_id: arxiv_2403_05798
arxiv_id: '2403.05798'
title: 'S2IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting'
authors: [Pan, Jiang, Garg, Schneider, Nevmyvaka, Song]
---

# S2IP-LLM: Semantic Space Informed Prompt Learning...

## Abstract

...
```

#### Чанкинг одной статьи

```python
from src.chunker.chunker import chunk_article
from src.common.io_utils import read_text

text = read_text("articles/2403_05798.md")
metadata, chunks = chunk_article(text, filename="2403_05798.md")

print(f"Article: {metadata.title}")
print(f"Chunks:  {len(chunks)}")
print()

for chunk in chunks[:3]:
    print(f"[{chunk.chunk_id}] section={chunk.section}, tokens={chunk.token_count}")
    print(f"  {chunk.text[:100]}...")
    print()
```

Результат:
```
Article: S2IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting
Chunks:  49

[arxiv_2403_05798_chunk_000] section=Unknown, tokens=52
  # S2IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting...

[arxiv_2403_05798_chunk_001] section=Abstract, tokens=198
  ## Abstract

  Recently, there has been a growing interest in leveraging...

[arxiv_2403_05798_chunk_002] section=Introduction, tokens=578
  ## 1. Introduction

  Time series forecasting is a critical task...
```

#### Парсинг блочной структуры

```python
from src.chunker.structure_parser import parse_blocks

text = open("articles/2403_05798.md").read()
blocks = parse_blocks(text)

for b in blocks[:8]:
    extra = f" (level={b.heading_level})" if b.heading_level else ""
    print(f"  {b.type:15s}{extra}  {b.content[:60]}")
```

Результат:
```
  heading        (level=1)  # S2IP-LLM: Semantic Space Informed Prompt Learni
  heading        (level=2)  ## Abstract
  paragraph                 Recently, there has been a growing interest in lev
  heading        (level=2)  ## 1. Introduction
  paragraph                 Time series forecasting is a critical task across
  paragraph                 Large Language Models (LLMs) have demonstrated rem
  heading        (level=3)  ### 1.1. Background
  paragraph                 Prior methods for time series forecasting can be b
```

#### Классификация заголовков

```python
from src.chunker.section_mapper import classify_heading

examples = [
    "## 1. Introduction",
    "## 2. Related Work",
    "#### III. METHODOLOGY",
    "#### A. Background",
    "## B. Long-term Forecasting Results",
    "## Acknowledgments",
]

for h in examples:
    label = classify_heading(h)
    print(f"  {h:45s} -> {label}")
```

Результат:
```
  ## 1. Introduction                           -> Introduction
  ## 2. Related Work                           -> RelatedWork
  #### III. METHODOLOGY                        -> Methodology
  #### A. Background                           -> Appendix_Implementation
  ## B. Long-term Forecasting Results          -> Appendix_Results
  ## Acknowledgments                           -> Unknown
```

#### Маскирование формул

```python
from src.chunker.math_masker import mask_formulas, unmask_formulas

text = r"Given $\mathbf{X} \in \mathbb{R}^{N \times D}$, the loss is $$\mathcal{L} = \sum_i \|y_i - \hat{y}_i\|^2$$"

masked, mapping = mask_formulas(text)
print("Masked:", masked)
print("Mapping:", mapping)

restored = unmask_formulas(masked, mapping)
print("Restored:", restored)
```

Результат:
```
Masked: Given [MATH_000], the loss is [DISPLAY_MATH_000]
Mapping: {
  '[DISPLAY_MATH_000]': '$$\\mathcal{L} = \\sum_i \\|y_i - \\hat{y}_i\\|^2$$',
  '[MATH_000]': '$\\mathbf{X} \\in \\mathbb{R}^{N \\times D}$'
}
Restored: Given $\mathbf{X} \in \mathbb{R}^{N \times D}$, the loss is $$\mathcal{L} = \sum_i \|y_i - \hat{y}_i\|^2$$
```

### Структура выходного JSON

Файл `chunks/per_article/2403_05798.json` (фрагмент):

```json
[
  {
    "chunk_id": "arxiv_2403_05798_chunk_000",
    "article_id": "arxiv_2403_05798",
    "section": "Unknown",
    "index": 0,
    "text": "# S2IP-LLM: Semantic Space Informed Prompt Learning...",
    "token_count": 52
  },
  {
    "chunk_id": "arxiv_2403_05798_chunk_001",
    "article_id": "arxiv_2403_05798",
    "section": "Abstract",
    "index": 1,
    "text": "## Abstract\n\nRecently, there has been a growing interest...",
    "token_count": 198
  }
]
```

Файл `chunks/dic_chunks.json` (обёртка):

```json
{
  "chunker_version": "0.1.0",
  "config_hash": "a1b2c3d4e5f6",
  "chunks": [ ... ]
}
```

---

## 7. Тестирование

```bash
# Все тесты
pytest

# С покрытием
pytest --cov=src --cov-report=term-missing

# Конкретный модуль
pytest tests/chunker/test_chunker.py -v
```

Тестовое покрытие: **81 тест** по всем модулям:

| Модуль | Файл тестов | Что покрывается |
|--------|-------------|-----------------|
| `artifact_cleaner` | `test_artifact_cleaner.py` | Image placeholders, emails, аффилиации, пустые строки таблиц |
| `section_remover` | `test_section_remover.py` | Удаление References, Impact Statement, вложенные секции |
| `header_normalizer` | `test_header_normalizer.py` | Склеенные заголовки, строка авторов |
| `math_normalizer` | `test_math_normalizer.py` | Разорванные формулы, standalone номера, ложные разрывы |
| `frontmatter_builder` | `test_frontmatter_builder.py` | article_id, arxiv_id, нормализация точек |
| `structure_parser` | `test_structure_parser.py` | Все типы блоков, frontmatter skip, multiline |
| `math_masker` | `test_math_masker.py` | Display/inline маскирование, roundtrip |
| `recursive_splitter` | `test_recursive_splitter.py` | Атомарность, merge, overlap, граничные случаи |
| `section_mapper` | `test_section_mapper.py` | Арабские и римские номера, IEEE-формат, appendix |
| `chunker` | `test_chunker.py` | Adaptive heading detection, flat/multi-level, group_by_section |
