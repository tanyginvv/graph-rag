# Lag-Llama: План семантического чанкинга (ревизия v1.2)

**article_id (Eval API):** `lag-llama`
**Вопросов Eval API:** 12 (q00–q11), типы А, А+, В, З, Г, Б, Д, Е, И, Ж, К, Л
**Дата ревизии:** 2026-05-22
**Основание ревизии:** переход от v1.0 (2 типа: А, Б с неверными индексами) к v1.2 (12 типов, стандартный порядок); q09=Ж без процентного улучшения

---

## 1. Актуальное соответствие «question_id → тип → раздел-источник»

Из `eval-api/docs/questions.md`, группа 1 (стандартный порядок, примечание ⁴: q09 без улучшения %):

| question_id | Тип | Семантика | Раздел-источник |
|-------------|-----|-----------|-----------------|
| `lag-llama_q00` | **А** | Все способы применения LLM / больших моделей | Abstract (18–20), 1.Introduction (26–36), 2.Related Work (44–45: LLM-based methods), 8.Discussion (233–237) |
| `lag-llama_q01` | **А+** | Наилучшие способы (по результатам) | 6.1 Zero-shot + Finetuning results (207–214, Табл. 1): avg rank 2.786 after finetuning |
| `lag-llama_q02` | **В** | Бенчмарки и наборы данных | 5.1 Datasets (138–143): 27 datasets, 6 domains; A. Tables 3–4 |
| `lag-llama_q03` | **З** | Метрики оценки качества | 5.4 Inference and Model Evaluation (178–181): CRPS, Average Rank |
| `lag-llama_q04` | **Г** | Представление TS / токенизация | 4.1 Tokenization: Lag Features (89–95): lag indices + date-time features |
| `lag-llama_q05` | **Б** | Способы улучшения прогноза | Lag features (89–95), Robust Standardization (112–130), Stratified Sampling + Freq-Mix/Freq-Mask (132–134), Student's t-distribution (108–110), Few-shot finetuning (215–219) |
| `lag-llama_q06` | **Д** | Нейросетевая архитектура / backbone | 4.2 Lag-Llama Architecture (96–106): decoder-only LLaMA-based, RMSNorm, RoPE, distribution head |
| `lag-llama_q07` | **Е** | Пайплайн и шаги прогнозирования | 4.2 Architecture (96–106): tokenization → linear proj → M decoder layers → distribution head → autoregressive decoding |
| `lag-llama_q08` | **И** | Подготовка данных и обучение доп. параметров | 4.4 Value Scaling (112–130), 4.5 Training Strategies (132–134), 5.3 Hyperparameter Search (174–176) |
| `lag-llama_q09` | **Ж** | Тип задачи, горизонт (без улучшения %) | Univariate probabilistic forecasting; variable prediction lengths; zero-shot/few-shot/finetune settings |
| `lag-llama_q10` | **К** | Предобучение/дообучение | Pretrained from scratch on 352M tokens; finetuned on downstream; few-shot adaptation |
| `lag-llama_q11` | **Л** | Нормировка / преобразование TS | 4.4 Value Scaling: Robust Standardization (median + IQR), summary statistics as covariates |

---

## 2. План чанков по типам вопросов

### Тип А: Все способы применения LLM / больших моделей (`lag-llama_q00`)

**Специфика:** Lag-Llama — foundation model, обученная с нуля (не LLM-адаптация). Для полного ответа нужны: (1) собственный подход Lag-Llama, (2) LLM-методы из Related Work.

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C01 | Abstract | 18–20 | prose | ~120 токенов |
| C02 | 1. Introduction: foundation model paradigm + contributions | 26–36 | prose | ~300 токенов |
| C03 | 2. Related Work: LLM-based methods (Time-LLM, LLM4TS, GPT2(6), UniTime, TEMPO) | 44–45 | prose | ~150 токенов |
| C04 | 8. Discussion | 233–237 | prose | ~200 токенов |

### Тип А+: Наилучшие способы (`lag-llama_q01`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C05 | 1. Introduction: contributions (items 3–4: SOTA after finetuning) | 34–35 | prose | ~100 токенов |
| C06 | 6.1 Results: Lag-Llama finetuned avg rank 2.786, best general-purpose | 207–214 | prose + Табл. 1 | ~500 токенов |

### Тип В: Бенчмарки и датасеты (`lag-llama_q02`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C07 | 5.1 Datasets: 27 datasets, 6 domains, 352M tokens | 138–143 | prose | ~200 токенов |
| C08 | A. Table 3 (Domains) + Table 4 (Statistics) | 573–638 | table_with_prose | ~800 токенов |

### Тип З: Метрики оценки (`lag-llama_q03`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C09 | 5.4 Inference and Model Evaluation: CRPS, Average Rank | 178–181 | prose | ~180 токенов |

### Тип Г: Представление TS (`lag-llama_q04`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C10 | 4.1 Tokenization: Lag Features (lag indices + date-time features) | 89–95 | prose | ~250 токенов |

### Тип Б: Способы улучшения прогноза (`lag-llama_q05`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C10 | 4.1 Lag Features (tokenization improvement) | 89–95 | prose | ~250 токенов |
| C11 | 4.4 Value Scaling: Robust Standardization (median + IQR) | 112–130 | prose + формулы | ~350 токенов |
| C12 | 4.5 Training Strategies: stratified sampling, Freq-Mix, Freq-Mask | 132–134 | prose | ~100 токенов |
| C13 | 4.3 Choice of Distribution Head: Student's t-distribution | 108–110 | prose | ~120 токенов |
| C14 | 6.2 Few-Shot Adaptation | 215–219 | prose | ~200 токенов |

**Переиспользование:** C10 общий с Г.

### Тип Д: Нейросетевая архитектура / backbone (`lag-llama_q06`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C15 | 4.2 Lag-Llama Architecture: decoder-only LLaMA, RMSNorm, RoPE | 96–106 | prose | ~300 токенов |
| C13 | 4.3 Distribution Head: Student's t-distribution | 108–110 | prose | ~120 токенов |
| C16 | D. Table 5 (Hyperparameters) | 639–653 | table | ~350 токенов |

**Переиспользование:** C13 общий с Б.

### Тип Е: Пайплайн и шаги прогнозирования (`lag-llama_q07`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C15 | 4.2 Architecture: tokenization → linear proj → M decoder layers → distribution head → autoregressive decoding | 96–106 | prose | ~300 токенов |

**Переиспользование:** C15 общий с Д.

### Тип И: Подготовка данных и обучение доп. параметров (`lag-llama_q08`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C11 | 4.4 Value Scaling: robust standardization + summary statistics | 112–130 | prose + формулы | ~350 токенов |
| C12 | 4.5 Training Strategies: stratified sampling, augmentations | 132–134 | prose | ~100 токенов |
| C17 | 5.3 Hyperparameter Search and Model Training | 174–176 | prose | ~150 токенов |

**Переиспользование:** C11 общий с Б и Л, C12 общий с Б.

### Тип Ж: Тип задачи, горизонт (`lag-llama_q09`)

**Примечание:** формулировка без процентного улучшения — «Каков тип задачи и горизонт прогнозирования».

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C18 | Abstract + 1.Introduction: univariate probabilistic forecasting | 18–20, 26–28 | prose | ~200 токенов |
| C19 | 3. Probabilistic Time Series Forecasting (формальная постановка) | 46–84 | formula_heavy | ~300 токенов |
| C07 | 5.1 Datasets: variable prediction lengths | 140–143 (фрагмент) | prose | ~100 токенов |

**Переиспользование:** C07 общий с В.

### Тип К: Предобучение/дообучение (`lag-llama_q10`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C20 | 1.Introduction: pretrained from scratch, diverse corpus | 32–33 | prose | ~100 токенов |
| C21 | 6.1 Zero-shot + Finetuning setup | 184–191 | prose | ~250 токенов |
| C17 | 5.3 Training setups: pretraining vs finetuning | 174–176 | prose | ~150 токенов |

**Переиспользование:** C17 общий с И.

### Тип Л: Нормировка / преобразование TS (`lag-llama_q11`)

| Чанк | Раздел | Строки | Тип | Размер |
|-------|--------|--------|-----|--------|
| C11 | 4.4 Value Scaling: Robust Standardization (median + IQR) | 112–130 | prose + формулы | ~350 токенов |
| C22 | 4.4 Value Scaling: standard scaler + summary statistics | 114–116 | prose + формулы | ~100 токенов |

**Переиспользование:** C11 общий с Б и И.

---

## 3. Итоговая таблица чанков (retrieval-слой)

| ID | Раздел | Строки | Типы вопросов | Приоритет |
|----|--------|--------|---------------|-----------|
| C01 | Abstract | 18–20 | А, Ж | Высокий |
| C02 | 1. Introduction (foundation model + contributions) | 26–36 | А, Ж | Высокий |
| C03 | 2. Related Work (LLM-based methods) | 44–45 | А | Высокий |
| C04 | 8. Discussion | 233–237 | А | Средний |
| C05 | 1. Introduction (SOTA after finetuning) | 34–35 | А+ | Средний |
| C06 | 6.1 Results (Табл. 1 + текст) | 207–214 | А+ | Высокий |
| C07 | 5.1 Datasets (текст) | 138–143 | В, Ж | Высокий |
| C08 | A. Tables 3–4 (Domains + Statistics) | 573–638 | В | Высокий |
| C09 | 5.4 Inference and Model Evaluation | 178–181 | З | Высокий |
| C10 | 4.1 Tokenization: Lag Features | 89–95 | Г, Б | Высокий |
| C11 | 4.4 Value Scaling (Robust Standardization) | 112–130 | Б, И, Л | Высокий |
| C12 | 4.5 Training Strategies | 132–134 | Б, И | Высокий |
| C13 | 4.3 Distribution Head (Student's t) | 108–110 | Б, Д | Средний |
| C14 | 6.2 Few-Shot Adaptation | 215–219 | Б | Высокий |
| C15 | 4.2 Lag-Llama Architecture | 96–106 | Д, Е | Высокий |
| C16 | D. Table 5 (Hyperparameters) | 639–653 | Д | Средний |
| C17 | 5.3 Hyperparameter Search + Training | 174–176 | И, К | Средний |
| C18 | Abstract + 1.Introduction (univariate prob. forecasting) | 18–20, 26–28 | Ж | Средний |
| C19 | 3. Probabilistic TS Forecasting (formulation) | 46–84 | Ж | Средний |
| C20 | 1.Introduction (pretrained from scratch) | 32–33 | К | Средний |
| C21 | 6.1 Zero-shot + Finetuning setup | 184–191 | К | Средний |
| C22 | 4.4 Value Scaling (standard scaler) | 114–116 | Л | Средний |

**Всего чанков:** 22 (против 12 в v1.0). Переиспользуемых: C01 (А+Ж), C02 (А+Ж), C07 (В+Ж), C10 (Г+Б), C11 (Б+И+Л), C12 (Б+И), C13 (Б+Д), C15 (Д+Е), C17 (И+К).

---

## 4. Переиспользование чанков между типами

9 переиспользуемых из 22. Наиболее нагруженный: C11 (3 типа: Б, И, Л).

---

## 5. Принципы и ограничения

1. **Дословность:** все чанки — verbatim-цитаты из markdown-версии.
2. **Foundation model, не LLM-адаптация:** для типа А обязательно включать C03 (LLM-методы из Related Work), так как собственная модель Lag-Llama — не LLM-based.
3. **q09=Ж без процентного улучшения:** вопрос только о типе задачи и горизонтах, без численных результатов.
4. **Robust Standardization (C11)** — ключевой чанк для трёх типов (Б, И, Л).
5. **Не чанкируются:** References (279–453), Contributions (245–271), Acknowledgements (273–277), дублирующие таблицы Appendix C.
