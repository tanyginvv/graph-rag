# TIME-LLM: План семантического чанкинга (ревизия v1.2)

**article_id (Eval API):** `time-llm`
**Вопросов Eval API:** 13 (q00–q12), типы А, А+, В, З, Г, Б, Д, Е, И, Ж, К, Л, М
**Дата ревизии:** 2026-05-22
**Основание ревизии:** переход от v1.0 (2 типа: А, Д с неверными индексами) к v1.2 (13 типов, стандартный порядок); единственная статья с типом М

---

## 1. Актуальное соответствие «question_id → тип → раздел-источник»

Из `eval-api/docs/questions.md`, группа 1 (стандартный порядок):

| question_id | Тип | Семантика | Раздел-источник |
|-------------|-----|-----------|-----------------|
| `time-llm_q00` | **А** | Все способы применения LLM | Abstract (24–27), 1.Introduction (28–51), 3.Methodology (64–110), 5.Conclusion (358–361) |
| `time-llm_q01` | **А+** | Наилучшие способы (по результатам) | 4.Main Results (111–357): 4.1–4.4 results + 4.5 Model Analysis/ablations |
| `time-llm_q02` | **В** | Бенчмарки и наборы данных | B.2 Dataset Details (500–526, Табл. 8) |
| `time-llm_q03` | **З** | Метрики оценки качества | B.3 Evaluation Metrics (528–556): MSE, MAE, SMAPE, MASE, OWA, MAPE |
| `time-llm_q04` | **Г** | Представление TS / токенизация | 3.1 Input Embedding (78–85): RevIN + patching + linear embed; Patch Reprogramming (86–98) |
| `time-llm_q05` | **Б** | Способы улучшения прогноза | Patch Reprogramming, Prompt-as-Prefix (101–108), RevIN, Text Prototypes, PaP components |
| `time-llm_q06` | **Д** | Нейросетевая архитектура / backbone | 3 (64–74), 3.1 (76–110): Llama-7B/GPT-2, Multi-head Cross-Attention, B.4 Табл. 9 |
| `time-llm_q07` | **Е** | Пайплайн и шаги прогнозирования | 3 Methodology (64–74): 5-step pipeline; 3.1 Model Structure (76–110) |
| `time-llm_q08` | **И** | Подготовка данных и обучение доп. параметров | 3.1 Input Embedding (78–85), B.1 Implementation (494–498), B.4 Табл. 9 |
| `time-llm_q09` | **Ж** | Тип задачи, горизонт, процентное улучшение | 4.1–4.4 Main Results: long-term/short-term/few-shot/zero-shot, horizons, improvement % |
| `time-llm_q10` | **К** | Предобучение/дообучение LLM | 3 (72): frozen LLM; 4.5 (348): 6.6M trainable params; QLoRA comparison |
| `time-llm_q11` | **Л** | Нормировка / преобразование TS | 3.1 RevIN (78), patching (78–84), B.1 scaling, trend/lag calculation (498) |
| `time-llm_q12` | **М** | Завершающий шаг прогноза | 3.1 Output Projection (109), B.1 flatten + linear projection (498) |

---

## 2. План чанков по типам вопросов

### Тип А: Все способы применения LLM (`time-llm_q00`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C01 | Abstract | 24–27 | prose | ~150 токенов |
| C02 | 1. Introduction: desiderata LLM + reprogramming concept | 32–44 | prose | ~350 токенов |
| C03 | 3. Methodology: общая схема (3 компонента) | 64–74 | prose + формулы | ~300 токенов |
| C04 | 3.1 Patch Reprogramming + Prompt-as-Prefix | 86–107 | prose | ~400 токенов |
| C05 | 5. Conclusion | 358–361 | prose | ~150 токенов |

### Тип А+: Наилучшие способы (`time-llm_q01`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C06 | 4. Main Results intro: TIME-LLM outperforms SOTA | 113 | prose | ~100 токенов |
| C07 | 4.5 Model Analysis: ablation results (Table 6 text) | 326–328 | prose | ~200 токенов |
| C08 | 4.1 Results text: 12% over GPT4TS, 20% over TimesNet | 121–122 | prose | ~150 токенов |

**Переиспользование:** C07 общий с Б, К; C08 общий с Ж.

### Тип В: Бенчмарки и датасеты (`time-llm_q02`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C09 | B.2 Dataset Details (текст) | 500–526 | prose | ~350 токенов |
| C10 | B.2 Табл. 8 (статистика датасетов) | 506–523 | table_with_prose | ~600 токенов |

### Тип З: Метрики оценки (`time-llm_q03`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C11 | B.3 Evaluation Metrics: MSE, MAE, SMAPE, MASE, OWA, MAPE | 528–556 | prose + формулы | ~500 токенов |

### Тип Г: Представление TS (`time-llm_q04`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C12 | 3.1 Input Embedding: RevIN + patching + linear embed | 78–85 | prose + формулы | ~250 токенов |
| C13 | 3.1 Patch Reprogramming: text prototypes E' + cross-attention | 86–98 | prose + формулы | ~350 токенов |
| C14 | 3.1 Prompt-as-Prefix: 3 компонента промпта | 101–108 | prose | ~200 токенов |

**Переиспользование:** C12 ⊂ C03, C13 ⊂ C04.

### Тип Б: Способы улучшения прогноза (`time-llm_q05`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C13 | 3.1 Patch Reprogramming (alignment improvement) | 86–98 | prose + формулы | ~350 токенов |
| C14 | 3.1 Prompt-as-Prefix (reasoning improvement) | 101–108 | prose | ~200 токенов |
| C15 | 3.1 RevIN (distribution shift) | 78 | prose | ~40 токенов |
| C07 | 4.5 Model Analysis: ablation confirms each component | 326–328 | prose | ~200 токенов |
| C16 | 4.5 Reprogramming Efficiency: lightweight (6.6M params) | 348 | prose | ~150 токенов |

**Переиспользование:** C13 общий с Г, C14 общий с Г, C15 ⊂ C12, C07 общий с А+ и К.

### Тип Д: Нейросетевая архитектура / backbone (`time-llm_q06`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C03 | 3. Methodology: архитектура (Llama, GPT-2) | 64–74 | prose + формулы | ~300 токенов |
| C17 | 3.1 Input Embedding (полный) | 78–85 | prose + формулы | ~250 токенов |
| C13 | 3.1 Patch Reprogramming: multi-head cross-attention | 86–98 | prose + формулы | ~350 токенов |
| C18 | 4.5 Language Model Variants: Llama(32/8), GPT-2(12/6) | 326–327 | prose | ~150 токенов |
| C19 | B.4 Model Configurations (Табл. 9) | 564–578 | table_with_prose | ~400 токенов |

**Переиспользование:** C03 общий с А, C13 общий с Г и Б.

### Тип Е: Пайплайн и шаги прогнозирования (`time-llm_q07`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C03 | 3. Methodology: 3 компонента + 5 шагов | 64–74 | prose + формулы | ~300 токенов |
| C20 | 3.1 Model Structure: 5 steps detailed | 76–110 | prose + формулы | ~500 токенов |

**Переиспользование:** C03 общий с А и Д.

### Тип И: Подготовка данных и обучение доп. параметров (`time-llm_q08`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C12 | 3.1 Input Embedding: RevIN + patching | 78–85 | prose + формулы | ~250 токенов |
| C21 | B.1 Implementation: text prototype learning, trend/lag | 498 | prose | ~200 токенов |
| C22 | 3 (72): frozen LLM, only input transformation + output projection trained | 72 | prose | ~100 токенов |
| C19 | B.4 Model Configurations (Табл. 9): LR, batch size, epochs | 564–578 | table_with_prose | ~400 токенов |

**Переиспользование:** C12 общий с Г, C19 общий с Д.

### Тип Ж: Тип задачи, горизонт, улучшение (`time-llm_q09`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C23 | 4.1 Long-term Forecasting: setups + horizons | 119–120 | prose | ~100 токенов |
| C08 | 4.1 Results text: improvement % | 121–122 | prose | ~150 токенов |
| C24 | 4.2 Short-term Forecasting: M4, M3-Quarterly | 170–174 | prose | ~150 токенов |
| C25 | 4.3 Few-shot: 10%/5% training data | 184–191 | prose | ~200 токенов |
| C26 | 4.4 Zero-shot: cross-domain | 284–288 | prose | ~200 токенов |
| C27 | 4.1 Табл. 1 (long-term results) | 125–167 | table_with_prose | ~600 токенов |

**Переиспользование:** C08 общий с А+.

### Тип К: Предобучение/дообучение LLM (`time-llm_q10`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C22 | 3 (72): frozen LLM, lightweight params trained | 72 | prose | ~100 токенов |
| C16 | 4.5 Reprogramming Efficiency: 6.6M params (0.2% of Llama-7B) | 348 | prose | ~150 токенов |
| C07 | 4.5 Model Analysis: scaling law preserved | 326–328 | prose | ~200 токенов |
| C28 | 3.1 Output Projection (109): flatten + linear | 109 | prose | ~50 токенов |

**Переиспользование:** C22 общий с И, C16 общий с Б, C07 общий с А+ и Б.

### Тип Л: Нормировка / преобразование TS (`time-llm_q11`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C15 | 3.1 RevIN normalization | 78 | prose | ~40 токенов |
| C29 | 3.1 Patching: деление TS на патчи | 78–84 | prose + формулы | ~200 токенов |
| C30 | B.1 Trend calculation + top-5 lag detection | 498 | prose | ~150 токенов |

**Переиспользование:** C15 общий с Б, C29 ⊂ C12.

### Тип М: Завершающий шаг прогноза (`time-llm_q12`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C28 | 3.1 Output Projection: flatten + linear → forecasts | 109 | prose | ~50 токенов |
| C31 | B.1 Output Projection: flatten P×D → 1D → linear → Ŷ ∈ R^H | 498 (фрагмент) | prose | ~100 токенов |

**Переиспользование:** C28 общий с К.

---

## 3. Итоговая таблица чанков (retrieval-слой)

| ID | Раздел | Строки | Типы вопросов | Приоритет |
|----|--------|--------|---------------|-----------|
| C01 | Abstract | 24–27 | А | Высокий |
| C02 | 1. Introduction (desiderata + reprogramming) | 32–44 | А | Высокий |
| C03 | 3. Methodology (архитектура + пайплайн) | 64–74 | А, Д, Е | Высокий |
| C04 | 3.1 Patch Reprogramming + PaP | 86–107 | А | Высокий |
| C05 | 5. Conclusion | 358–361 | А | Средний |
| C06 | 4. Main Results intro (SOTA) | 113 | А+ | Средний |
| C07 | 4.5 Model Analysis (ablations text) | 326–328 | А+, Б, К | Высокий |
| C08 | 4.1 Results text (improvement %) | 121–122 | А+, Ж | Высокий |
| C09 | B.2 Dataset Details (текст) | 500–526 | В | Высокий |
| C10 | B.2 Табл. 8 | 506–523 | В | Высокий |
| C11 | B.3 Evaluation Metrics | 528–556 | З | Высокий |
| C12 | 3.1 Input Embedding (RevIN + patching) | 78–85 | Г, И | Высокий |
| C13 | 3.1 Patch Reprogramming (cross-attention) | 86–98 | Г, Б, Д | Высокий |
| C14 | 3.1 Prompt-as-Prefix (3 компонента) | 101–108 | Г, Б | Высокий |
| C15 | 3.1 RevIN (нормировка) | 78 | Б, Л | Средний |
| C16 | 4.5 Reprogramming Efficiency (6.6M) | 348 | Б, К | Средний |
| C17 | 3.1 Input Embedding (полный) | 78–85 | Д | Средний |
| C18 | 4.5 Language Model Variants | 326–327 | Д | Средний |
| C19 | B.4 Model Configurations (Табл. 9) | 564–578 | Д, И | Средний |
| C20 | 3.1 Model Structure (5 steps) | 76–110 | Е | Средний |
| C21 | B.1 Implementation (text prototypes, trend/lag) | 498 | И | Средний |
| C22 | 3 (72): frozen LLM, trainable params | 72 | И, К | Высокий |
| C23 | 4.1 Long-term setups + horizons | 119–120 | Ж | Высокий |
| C24 | 4.2 Short-term forecasting | 170–174 | Ж | Средний |
| C25 | 4.3 Few-shot (10%/5%) | 184–191 | Ж | Средний |
| C26 | 4.4 Zero-shot (cross-domain) | 284–288 | Ж | Средний |
| C27 | 4.1 Табл. 1 (long-term results) | 125–167 | Ж | Высокий |
| C28 | 3.1 Output Projection | 109 | К, М | Средний |
| C29 | 3.1 Patching (сегментация) | 78–84 | Л | Средний |
| C30 | B.1 Trend + top-5 lags | 498 | Л | Средний |
| C31 | B.1 Output Projection (flatten + linear) | 498 | М | Средний |

**Всего чанков:** 31 (против 12 в v1.0). Переиспользуемых: C03 (А+Д+Е), C07 (А++Б+К), C08 (А++Ж), C12 (Г+И), C13 (Г+Б+Д), C14 (Г+Б), C15 (Б+Л), C16 (Б+К), C22 (И+К), C28 (К+М).

---

## 4. Переиспользование чанков между типами

10 переиспользуемых чанков из 31. Наиболее нагруженные: C03 (3 типа), C07 (3 типа), C13 (3 типа).

---

## 5. Принципы и ограничения

1. **Дословность:** все чанки — verbatim-цитаты из markdown-версии (с сохранением LaTeX-формул и HTML-таблиц).
2. **Единственная статья с типом М:** завершающий шаг прогноза (Output Projection) — уникальный вопрос для Time-LLM.
3. **Табл. 1 (long-term results)** — крупная (~600 токенов), но логически неделимая; обслуживает типы Ж и А+.
4. **Раздел 4.5 Model Analysis** — ключевой для А+ (абляции), Б (подтверждение улучшений), К (scaling law).
5. **Не чанкируются:** References (362–481), Figure-описания, дублирующие таблицы Приложений D–E, раздел A (More Related Work — дублирует раздел 2).
