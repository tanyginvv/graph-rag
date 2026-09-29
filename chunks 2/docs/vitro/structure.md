# VITRO: Структурная разметка

**article_id (Eval API):** `vitro`
**Файл:** `VITRO Vocabulary Inversion for Time-series Representation Optimization.md`
**Всего строк:** 392

---

## Иерархия разделов

| Уровень | Номер | Название раздела | Строки в MD | Примечания |
|---------|-------|-----------------|-------------|------------|
| — | — | Титул + авторы + аффилиации | 1–19 | 3 автора, UMich + Capital One |
| 0 | — | **Abstract** | 20–21 | Аннотация: vocabulary inversion |
| — | — | Index Terms | 22 | Ключевые слова |
| 1 | I | **Introduction** | 24–38 | Введение: проблема vocabulary gap + обзор методов |
| 1 | II | **Method** | 40–167 | Метод: двухстадийный процесс |
| 2 | II.A | Problem Formulation and Overview | 42–54 | Формальная постановка + два этапа |
| 2 | II.B | Stage 1: Vocabulary Inversion for Time Series | 55–130 | Этап 1: обучение псевдослов v_i и s |
| 3 | — | Model Pipeline | 70–98 | Пайплайн: RevIN → patching → embedding → concatenation → frozen LLM → projection |
| 3 | — | Optimization Objective | 100–129 | Формулы (2) и (3): минимизация MSE |
| 2 | II.C | Stage 2: Time Series Forecasting with Learned Vocabulary | 131–167 | Этап 2: два подхода |
| 3 | — | Similarity-based Selection (Sim) | 135–155 | Core lexicon C → cosine similarity → top-k selection (backbone: GPT-2) |
| 3 | — | Attention-based approach | 157–167 | Multi-head cross-attention (backbone: Llama-7B, на базе TimeLLM) |
| 1 | III | **Experiments** | 169–334 | Эксперименты |
| 2 | III.A | Long-term Forecasting | 175–334 | Табл. I: результаты на 7 датасетах |
| 1 | IV | **Qualitative Analysis** | 336–344 | Качественный анализ: PCA, t-SNE, attention heatmaps |
| 1 | V | **Conclusion and Future Work** | 346–348 | Заключение |
| 1 | — | **References** | 350–392 | Список литературы |

---

## Ключевые структурные блоки (для чанкинга) — ревизия v1.2

| Блок | Разделы | Тип содержимого | Релевантные типы вопросов |
|------|---------|-----------------|--------------------------|
| **Аннотация** | Abstract | prose | А |
| **Введение + проблема vocabulary gap** | I. Introduction | prose | А |
| **Метод: обзор двух этапов** | II.A | prose + формулы | А, **Е**, **И** |
| **Метод: Stage 1 (vocabulary inversion)** | II.B | prose + формулы | А, **Г**, **И**, **К** |
| **Метод: Stage 2 — Sim approach** | II.C (Similarity-based) | prose + формулы | А, **Д**, **Б** |
| **Метод: Stage 2 — Attention approach** | II.C (Attention-based) | prose + формулы | А, **Д** |
| **Эксперименты: Setup + Baselines** | III (начало) | prose | А, **В**, **З**, **Ж** |
| **Результаты: Табл. I** | III.A | table_with_prose | **В**, **Ж**, **А+** |
| **Результаты: текст** | III.A (ст. 334) | prose | **Б**, **Ж**, **А+** |
| **Качественный анализ** | IV | prose | — (3.3) |
| **Заключение** | V | prose | А |

**Жирным** выделены типы, добавленные в ревизии v1.2 (9 типов: Г, Д, Е, З, Б, И, Ж, А+, К).

---

## Примечания

1. **VITRO — метод улучшения представления TS:** основная идея — замена general-purpose LLM vocabulary на специализированный time-series vocabulary через vocabulary inversion (textual inversion из vision-language домена).
2. **Двухстадийный процесс:** Stage 1 оптимизирует псевдослова (v_i для каждого экземпляра + s для датасета), Stage 2 использует их для прогнозирования.
3. **Два LLM-backbone:** GPT-2 (Similarity-based Selection) и Llama-7B (Attention-based, на базе TimeLLM). Обе frozen.
4. **11 типов вопросов (v1.2):** А, Г, Д, Е, В, З, Б, И, Ж, А+, К. Старая версия (v1.0) покрывала только А и В.
5. **Альтернативный порядок индексов:** q01=Г, q02=Д, q04=В и т.д. — всегда ориентироваться на `question_id`, а не на индекс.
6. **Таблица I** — единственная таблица результатов: 7 датасетов × 7 методов, усреднённые MSE/MAE по 4 горизонтам. Обслуживает типы В, Ж, А+.
7. **Раздел III.A Results (ст. 334)** — ключевой для типов Б (улучшения), Ж (итоги), А+ (наилучшие методы).
8. **Индексные ссылки:** статья использует числовые ссылки [1]–[20] вместо автор-год.
9. **Рисунки 1–3** — текстовые описания, не содержат retrieval-данных.
