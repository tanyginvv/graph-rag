# VITRO: План семантического чанкинга (ревизия v1.2)

**article_id (Eval API):** `vitro`
**Вопросов Eval API:** 11 (q00–q10), типы А, Г, Д, Е, В, З, Б, И, Ж, А+, К
**Дата ревизии:** 2026-05-22
**Основание ревизии:** переход от v1.0 (2 типа: А, В) к v1.2 (11 типов); альтернативный порядок индексов (группа 2)

---

## 1. Актуальное соответствие «question_id → тип → раздел-источник»

Из `eval-api/docs/questions.md`, группа 2 (vitro, ts-rag):

| question_id | Тип | Семантика | Раздел-источник |
|-------------|-----|-----------|-----------------|
| `vitro_q00` | **А** | Все способы применения LLM | Abstract (20–21), I.Introduction (24–38), II.A (42–54), II.B (55–130), II.C (131–167), V.Conclusion (346–348) |
| `vitro_q01` | **Г** | Представление TS / токенизация | II.B: patching (70–78), patch embeddings (78–80), pseudo-word embeddings v_i + s (55–68) |
| `vitro_q02` | **Д** | Нейросетевая архитектура / backbone | II.C: GPT-2 (Sim, ст. 155), Llama-7B (Attention, ст. 167); обе frozen |
| `vitro_q03` | **Е** | Пайплайн и шаги прогнозирования | II.A (42–54): two-stage overview; II.B Model Pipeline (70–98): RevIN → patching → embedding → concatenation → frozen LLM → projection |
| `vitro_q04` | **В** | Бенчмарки и наборы данных | III.A Setup (175–177): 7 datasets; Табл. I (181–332) |
| `vitro_q05` | **З** | Метрики оценки качества | III.A Setup (ст. 177): MSE, MAE |
| `vitro_q06` | **Б** | Способы улучшения прогноза | Vocabulary Inversion (замена general vocabulary), RevIN, Two-stage training, Core Lexicon Reduction, Iterative optimization |
| `vitro_q07` | **И** | Подготовка данных и обучение доп. параметров | II.B: RevIN (70), patching (70–78), linear patch embedding (78), e_stats (84); frozen LLM, trainable params |
| `vitro_q08` | **Ж** | Тип задачи, горизонт, процентное улучшение | III.A: long-term forecasting, horizons {96, 192, 336, 720}; Results (334): improvement summary; Табл. I: конкретные улучшения |
| `vitro_q09` | **А+** | Наилучшие способы (по результатам) | III.A Results (334): VITRO-enhanced consistently outperform; Табл. I: VITRO-Sim > Sim, VITRO-TimeLLM > TimeLLM |
| `vitro_q10` | **К** | Предобучение/дообучение LLM | II.B: frozen LLM strategy; trainable only: v_i, s, W_e, b_e, W, b; no LLM fine-tuning |

---

## 2. План чанков по типам вопросов

### Тип А: Все способы применения LLM (`vitro_q00`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C01 | Abstract | 20–21 | prose | ~100 токенов |
| C02 | I. Introduction (проблема + существующие методы + идея VITRO) | 26–34 | prose | ~350 токенов |
| C03 | II.A Problem Formulation and Overview | 42–54 | prose + формулы | ~300 токенов |
| C04 | II.B Stage 1: Vocabulary Inversion (ключевые абзацы) | 55–68 | prose | ~300 токенов |
| C05 | II.B Model Pipeline (RevIN → frozen LLM → projection) | 70–98 | prose + формулы | ~450 токенов |
| C06 | II.C Stage 2: Sim + Attention approaches | 131–167 | prose + формулы | ~500 токенов |
| C07 | V. Conclusion | 346–348 | prose | ~100 токенов |

Компактный вариант: C01 + C02 + C03 + C07 (~850 токенов).

### Тип Г: Представление TS (`vitro_q01`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C08 | II.B Patching + Patch Embeddings | 70–78 | prose + формулы | ~200 токенов |
| C09 | II.B Pseudo-word Embeddings (v_i, s) + Vocabulary Inversion | 55–68 | prose | ~300 токенов |
| C10 | II.B Patch Embeddings as Prompts | 80 | prose | ~80 токенов |

**Переиспользование:** C08 ⊂ C05, C09 ⊂ C04, C10 ⊂ C05.

### Тип Д: Нейросетевая архитектура / backbone (`vitro_q02`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C11 | II.C Sim approach: GPT-2 backbone | 135–155 | prose + формулы | ~250 токенов |
| C12 | II.C Attention approach: Llama-7B backbone + cross-attention | 157–167 | prose + формулы | ~200 токенов |

**Переиспользование:** C11 ⊂ C06, C12 ⊂ C06.

### Тип Е: Пайплайн и шаги прогнозирования (`vitro_q03`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C03 | II.A Problem Formulation and Overview (two-stage) | 42–54 | prose + формулы | ~300 токенов |
| C05 | II.B Model Pipeline (RevIN → patching → embedding → concatenation → frozen LLM → projection) | 70–98 | prose + формулы | ~450 токенов |
| C13 | II.C Stage 2 intro: two forecasting approaches | 131–134 | prose | ~100 токенов |

**Переиспользование:** C03 общий с А, C05 общий с А.

### Тип В: Бенчмарки и датасеты (`vitro_q04`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C14 | III.A Setup (7 datasets + horizons + metrics) | 175–177 | prose | ~150 токенов |
| C15 | III.A Табл. I (результаты) | 181–332 | table_with_prose | ~800 токенов |

### Тип З: Метрики оценки (`vitro_q05`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C16 | III.A Setup (MSE, MAE) | 177 (фрагмент) | prose | ~50 токенов |

**Переиспользование:** C16 ⊂ C14.

### Тип Б: Способы улучшения прогноза (`vitro_q06`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C17 | II.B Vocabulary Inversion: замена general vocabulary | 57–68 | prose | ~250 токенов |
| C18 | II.B RevIN (distribution shift mitigation) | 70 | prose | ~50 токенов |
| C19 | II.C Core Lexicon Reduction (efficiency) | 135–147 | prose + формулы | ~200 токенов |
| C20 | III.A Results: consistent improvements | 334 | prose | ~200 токенов |

**Переиспользование:** C17 ⊂ C04, C18 ⊂ C05, C19 ⊂ C11, C20 общий с Ж и А+.

### Тип И: Подготовка данных и обучение доп. параметров (`vitro_q07`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C18 | II.B RevIN | 70 | prose | ~50 токенов |
| C21 | II.B Patching + Linear Embedding + e_stats | 70–84 | prose + формулы | ~300 токенов |
| C22 | II.B Optimization Objective: trainable params, frozen LLM | 55–68, 121–129 | prose + формулы | ~250 токенов |
| C23 | II.A Two-stage process overview | 53–54 | prose | ~80 токенов |

**Переиспользование:** C18 общий с Б, C21 ⊂ C05, C22 ⊂ C04.

### Тип Ж: Тип задачи, горизонт, улучшение (`vitro_q08`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C14 | III.A Setup: long-term forecasting, horizons {96, 192, 336, 720} | 175–177 | prose | ~150 токенов |
| C20 | III.A Results: SOTA, consistent improvements | 334 | prose | ~200 токенов |
| C15 | III.A Табл. I (конкретные MSE/MAE улучшения) | 181–332 | table_with_prose | ~800 токенов |

**Переиспользование:** C14 общий с В, C20 общий с Б и А+, C15 общий с В.

### Тип А+: Наилучшие способы (`vitro_q09`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C20 | III.A Results: VITRO-enhanced consistently outperform, SOTA | 334 | prose | ~200 токенов |
| C24 | III.A Results: vs S2IP-LLM, PatchTST, DLinear | 334 (вторая половина) | prose | ~150 токенов |

**Переиспользование:** C20 общий с Б и Ж.

### Тип К: Предобучение/дообучение LLM (`vitro_q10`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C22 | II.B Optimization: trainable only v_i, s, embedder, projection; frozen LLM | 55–68, 121–129 | prose + формулы | ~250 токенов |
| C25 | II.C Stage 2: frozen LLM in both approaches | 155, 167 | prose | ~80 токенов |

**Переиспользование:** C22 общий с И.

---

## 3. Итоговая таблица чанков (retrieval-слой)

| ID | Раздел | Строки | Типы вопросов | Приоритет |
|----|--------|--------|---------------|-----------|
| C01 | Abstract | 20–21 | А | Высокий |
| C02 | I. Introduction | 26–34 | А | Высокий |
| C03 | II.A Problem Formulation (two-stage) | 42–54 | А, Е | Высокий |
| C04 | II.B Stage 1: Vocabulary Inversion | 55–68 | А | Высокий |
| C05 | II.B Model Pipeline | 70–98 | А, Е | Высокий |
| C06 | II.C Stage 2: Sim + Attention | 131–167 | А | Высокий |
| C07 | V. Conclusion | 346–348 | А | Средний |
| C08 | II.B Patching + Patch Embeddings | 70–78 | Г | Средний |
| C09 | II.B Pseudo-word Embeddings (v_i, s) | 55–68 | Г | Высокий |
| C10 | II.B Patch Embeddings as Prompts | 80 | Г | Низкий |
| C11 | II.C Sim approach: GPT-2 backbone | 135–155 | Д | Высокий |
| C12 | II.C Attention: Llama-7B + cross-attention | 157–167 | Д | Высокий |
| C13 | II.C Stage 2 intro (two approaches) | 131–134 | Е | Средний |
| C14 | III.A Setup (datasets + horizons + metrics) | 175–177 | В, Ж | Высокий |
| C15 | III.A Табл. I (результаты) | 181–332 | В, Ж | Высокий |
| C16 | III.A Setup (MSE, MAE) | 177 | З | Средний |
| C17 | II.B Vocabulary Inversion (улучшение) | 57–68 | Б | Высокий |
| C18 | II.B RevIN | 70 | Б, И | Средний |
| C19 | II.C Core Lexicon Reduction | 135–147 | Б | Средний |
| C20 | III.A Results (текст) | 334 | Б, Ж, А+ | Высокий |
| C21 | II.B Patching + Linear Embed + e_stats | 70–84 | И | Средний |
| C22 | II.B Optimization: frozen LLM, trainable params | 55–68, 121–129 | И, К | Высокий |
| C23 | II.A Two-stage process overview | 53–54 | И | Средний |
| C24 | III.A Results: сравнение с baselines | 334 | А+ | Средний |
| C25 | II.C Frozen LLM (Sim + Attention) | 155, 167 | К | Средний |

**Всего чанков:** 25 (против 9 в v1.0). Переиспользуемых: C03 (А+Е), C05 (А+Е), C14 (В+Ж), C15 (В+Ж), C18 (Б+И), C20 (Б+Ж+А+), C22 (И+К).

---

## 4. Переиспользование чанков между типами

| Чанк | А | Г | Д | Е | В | З | Б | И | Ж | А+ | К |
|------|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:--:|:-:|
| C03 | ✓ | | | ✓ | | | | | | | |
| C05 | ✓ | | | ✓ | | | | | | | |
| C14 | | | | | ✓ | | | | ✓ | | |
| C15 | | | | | ✓ | | | | ✓ | | |
| C18 | | | | | | | ✓ | ✓ | | | |
| C20 | | | | | | | ✓ | | ✓ | ✓ | |
| C22 | | | | | | | | ✓ | | | ✓ |

7 переиспользуемых чанков из 25 (было 0 в v1.0).

---

## 5. Принципы и ограничения

1. **Дословность:** все чанки — verbatim-цитаты из markdown-версии.
2. **Семантическая завершённость:** Stage 1 и Stage 2 — отдельные чанки, каждый самодостаточен.
3. **Альтернативный порядок индексов:** vitro использует нестандартный порядок (q01=Г, q02=Д, q04=В и т.д.) — всегда ориентироваться на `question_id`, а не на индекс.
4. **Табл. I — центральный артефакт:** обслуживает типы В (датасеты), Ж (результаты) и косвенно А+ (наилучшие методы). Крупная (~800 токенов), но логически неделимая.
5. **Раздел III.A Results (ст. 334)** — многоцелевой: покрывает Б (улучшения), Ж (итоги), А+ (наилучшие). При фрагментации стоит разделить на две части: общее описание улучшений и сравнение с baseline.
6. **Не чанкируются:** References (350–392), Qualitative Analysis (IV — визуализации), Figure-описания, Index Terms (22).
