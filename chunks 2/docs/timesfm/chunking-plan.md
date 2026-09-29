# TimesFM: План семантического чанкинга (ревизия v1.2)

**article_id (Eval API):** `timesfm`
**Вопросов Eval API:** 12 (q00–q11), типы А, А+, В, З, Г, Б, Д, Е, И, Ж, К, Л
**Дата ревизии:** 2026-05-22
**Основание ревизии:** переход от v1.0 (2 типа: А, Е с неверными индексами) к v1.2 (12 типов, стандартный порядок); q04=Г со спец. формулировкой³

---

## 1. Актуальное соответствие «question_id → тип → раздел-источник»

Из `eval-api/docs/questions.md`, группа 1 (примечание ³: q04=Г — «LLM-подобная модель для представления TS»):

| question_id | Тип | Семантика | Раздел-источник |
|-------------|-----|-----------|-----------------|
| `timesfm_q00` | **А** | Все способы применения LLM / больших моделей | Abstract (16–18), 1.Introduction (26–33), 2.Related Work (45), 4.Model Architecture (58–117), 7.Conclusion (215–217) |
| `timesfm_q01` | **А+** | Наилучшие способы (по результатам) | 6.1 Zero-shot Evaluation (166–186): top model Monash, 25% better than llmtime |
| `timesfm_q02` | **В** | Бенчмарки и наборы данных | 5 Pretraining (табл. 1: 19 источников); 6.1: Monash (18), Darts (8), ETT (4) |
| `timesfm_q03` | **З** | Метрики оценки качества | 6.1: MAE, Scaled MAE (GM), msMAPE; Appendix A.2 |
| `timesfm_q04` | **Г** | LLM-подобная модель для представления TS | 4: Patching (62–63), Input Residual Block (74–83), Positional Encoding (83), Longer output patches (66–69) |
| `timesfm_q05` | **Б** | Способы улучшения прогноза | Patching, Longer output patches, Patch Masking (70–71), Synthetic data (126–127), Autoregressive decoding (196–198) |
| `timesfm_q06` | **Д** | Нейросетевая архитектура / backbone | 4: Decoder-only (64–65), Stacked Transformer (85–92), causal attention, Residual Blocks |
| `timesfm_q07` | **Е** | Пайплайн и шаги прогнозирования | 4: Input → Residual Block → PE → Stacked Transformer → Output Residual Block → Autoregressive Decoding (74–115) |
| `timesfm_q08` | **И** | Подготовка данных и обучение доп. параметров | 4 Training (112–113): patch masking, standard normalization; 5: data mixing, synthetic data |
| `timesfm_q09` | **Ж** | Тип задачи, горизонт, процентное улучшение | 6.1: zero-shot forecasting, variable horizons; 25% over llmtime; Figure 2 results |
| `timesfm_q10` | **К** | Предобучение/дообучение LLM | 5 Pretraining: from scratch, 200M params; 6.2 Scaling study (192–194); Appendix A.3 finetuning |
| `timesfm_q11` | **Л** | Нормировка / преобразование TS | 5: standard normalization (context mean/std scaling of first input patch); Appendix A.4 |

---

## 2. План чанков по типам вопросов

### Тип А: Все способы применения (`timesfm_q00`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C01 | Abstract | 16–18 | prose | ~100 токенов |
| C02 | 1. Introduction: foundation model concept + TimesFM key elements | 26–33 | prose | ~300 токенов |
| C03 | 4. Model Architecture: guiding principles (patching, decoder-only, longer output patches) | 60–71 | prose | ~300 токенов |
| C04 | 7. Conclusion | 215–217 | prose | ~100 токенов |

### Тип А+: Наилучшие способы (`timesfm_q01`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C05 | 6.1 Zero-shot Evaluation: TimesFM top model on Monash, 25% better than llmtime | 176–177 | prose | ~150 токенов |
| C06 | 6.1: Darts + ETT results summary | 180–184 | prose | ~250 токенов |
| C07 | 6.2 Scaling study: errors decrease monotonically with FLOPS | 194 | prose | ~100 токенов |

### Тип В: Бенчмарки и датасеты (`timesfm_q02`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C08 | 6.1: Monash (18 datasets), Darts (8), ETT (4) | 168–169, 172–173, 178, 182–183 | prose | ~300 токенов |
| C09 | 5. Table 1 (pretraining data composition) | 132–154 | table_with_prose | ~600 токенов |

### Тип З: Метрики оценки (`timesfm_q03`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C10 | 6.1: MAE, Scaled MAE (GM), msMAPE | 174–175, 180 | prose | ~150 токенов |
| C11 | 4. Loss Function: MSE during training | 103–110 | prose + формулы | ~150 токенов |

### Тип Г: LLM-подобное представление TS (`timesfm_q04`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C12 | 4: Patching — token analogue for time-series | 62–63 | prose | ~100 токенов |
| C13 | 4: Input Layers (Residual Block + PE) | 74–83 | prose + формулы | ~250 токенов |
| C14 | 4: Longer output patches (h > p) | 66–69 | prose | ~150 токенов |

### Тип Б: Способы улучшения (`timesfm_q05`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C12 | 4: Patching (efficiency + performance) | 62–63 | prose | ~100 токенов |
| C14 | 4: Longer output patches (fewer AR steps) | 66–69 | prose | ~150 токенов |
| C15 | 4: Patch Masking strategy | 70–71 | prose | ~100 токенов |
| C16 | 6.2: Autoregressive Decoding ablation | 196–198 | prose | ~150 токенов |
| C17 | 6.2: Synthetic Data ablation | 204–205 | prose | ~150 токенов |

**Переиспользование:** C12 общий с Г, C14 общий с Г.

### Тип Д: Нейросетевая архитектура (`timesfm_q06`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C18 | 4: Decoder-only model | 64–65 | prose | ~100 токенов |
| C19 | 4: Stacked Transformer (causal attention, FFN) | 85–92 | prose + формулы | ~250 токенов |
| C13 | 4: Input Residual Block + Output Residual Block | 74–83, 94–99 | prose + формулы | ~300 токенов |

**Переиспользование:** C13 общий с Г и Е.

### Тип Е: Пайплайн и шаги прогнозирования (`timesfm_q07`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C03 | 4: Guiding principles + overall architecture | 60–72 | prose | ~300 токенов |
| C20 | 4: Full pipeline — Input → Transformer → Output → Loss → Training → Inference | 74–115 | prose + формулы | ~500 токенов |

**Переиспользование:** C03 общий с А.

### Тип И: Подготовка данных и обучение (`timesfm_q08`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C15 | 4: Patch Masking strategy (training) | 70–71, 112–113 | prose | ~150 токенов |
| C21 | 5: Standard normalization (context mean/std) | 130–131 | prose | ~100 токенов |
| C22 | 5: Dataset Mixing and Training (80/20 real/synthetic) | 130–131 | prose | ~150 токенов |

**Переиспользование:** C15 общий с Б.

### Тип Ж: Тип задачи, горизонт, улучшение (`timesfm_q09`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C23 | 3. Problem Definition: zero-shot forecaster, variable C and H | 47–56 | prose + формулы | ~200 токенов |
| C05 | 6.1: Monash — 25% over llmtime | 176–177 | prose | ~150 токенов |
| C24 | 6.1: Results summary for all 3 benchmark groups | 160, 168–184 | prose | ~300 токенов |

**Переиспользование:** C05 общий с А+.

### Тип К: Предобучение/дообучение (`timesfm_q10`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C25 | 1.Introduction: 200M params, O(100B) timepoints | 33 | prose | ~80 токенов |
| C26 | 5: Pretraining data composition + training setup | 118–131 | prose | ~400 токенов |
| C07 | 6.2: Scaling study (17M → 70M → 200M) | 192–194 | prose | ~100 токенов |

**Переиспользование:** C07 общий с А+.

### Тип Л: Нормировка / преобразование TS (`timesfm_q11`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C21 | 5: Standard normalization (context mean/std of first input patch) | 130–131 | prose | ~100 токенов |
| C27 | Appendix A.4: Reversible Instance Normalization (RevIN) reference | Appendix | prose | ~80 токенов |

**Переиспользование:** C21 общий с И.

---

## 3. Итоговая таблица чанков (retrieval-слой)

| ID | Раздел | Строки | Типы вопросов | Приоритет |
|----|--------|--------|---------------|-----------|
| C01 | Abstract | 16–18 | А | Высокий |
| C02 | 1. Introduction (foundation model + TimesFM) | 26–33 | А | Высокий |
| C03 | 4. Guiding principles + architecture overview | 60–72 | А, Е | Высокий |
| C04 | 7. Conclusion | 215–217 | А | Средний |
| C05 | 6.1 Monash results (top model, 25% over llmtime) | 176–177 | А+, Ж | Высокий |
| C06 | 6.1 Darts + ETT results | 180–184 | А+ | Средний |
| C07 | 6.2 Scaling study | 194 | А+, К | Средний |
| C08 | 6.1 Benchmark datasets (Monash, Darts, ETT) | 168–184 | В | Высокий |
| C09 | 5. Table 1 (pretraining data) | 132–154 | В | Высокий |
| C10 | 6.1 Metrics: MAE, Scaled MAE, msMAPE | 174–175, 180 | З | Высокий |
| C11 | 4. Loss Function (MSE) | 103–110 | З | Средний |
| C12 | 4. Patching (token analogue) | 62–63 | Г, Б | Высокий |
| C13 | 4. Input Layers (Residual Block + PE) | 74–83 | Г, Д | Высокий |
| C14 | 4. Longer output patches (h > p) | 66–69 | Г, Б | Высокий |
| C15 | 4. Patch Masking strategy | 70–71, 112–113 | Б, И | Высокий |
| C16 | 6.2 Autoregressive Decoding ablation | 196–198 | Б | Средний |
| C17 | 6.2 Synthetic Data ablation | 204–205 | Б | Средний |
| C18 | 4. Decoder-only model | 64–65 | Д | Высокий |
| C19 | 4. Stacked Transformer (causal attention) | 85–92 | Д | Высокий |
| C20 | 4. Full pipeline (training + inference) | 74–115 | Е | Высокий |
| C21 | 5. Standard normalization | 130–131 | И, Л | Средний |
| C22 | 5. Dataset Mixing and Training | 130–131 | И | Средний |
| C23 | 3. Problem Definition | 47–56 | Ж | Средний |
| C24 | 6.1 Results summary (3 benchmark groups) | 160, 168–184 | Ж | Средний |
| C25 | 1. Introduction (200M params, 100B timepoints) | 33 | К | Средний |
| C26 | 5. Pretraining details (data + training) | 118–131 | К | Высокий |
| C27 | Appendix A.4 (RevIN reference) | Appendix | Л | Низкий |

**Всего чанков:** 27 (против 10 в v1.0). Переиспользуемых: C03 (А+Е), C05 (А++Ж), C07 (А++К), C12 (Г+Б), C13 (Г+Д), C14 (Г+Б), C15 (Б+И), C21 (И+Л).

---

## 4. Принципы и ограничения

1. **Дословность:** все чанки — verbatim-цитаты из markdown-версии.
2. **Foundation model с нуля:** TimesFM — не LLM-адаптация; тип А требует отдельного внимания.
3. **q04=Г со спец. формулировкой:** «LLM-подобная модель для представления» — фокус на patching как аналог токенизации.
4. **Табл. 1 (C09)** — крупная (~600 токенов), логически неделимая.
5. **Не чанкируются:** References, Impact Statement (8), Figure-описания.
