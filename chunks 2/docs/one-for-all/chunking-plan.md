# One-for-All: План семантического чанкинга (ревизия v1.2)

**article_id:** `one-for-all` | **Вопросов:** 12 (q00–q11), типы А–Л | **Дата:** 2026-05-22

## 1. Актуальное соответствие

| question_id | Тип | Раздел-источник |
|-------------|-----|-----------------|
| `one-for-all_q00` | **А** | Abstract, I.Introduction, III.Methodology (rsLoRA + GPT-2) |
| `one-for-all_q01` | **А+** | IV.Results: efficiency-accuracy trade-offs |
| `one-for-all_q02` | **В** (расшир.) | IV: ETT, Weather, M3, M4 datasets |
| `one-for-all_q03` | **З** | MSE, MAE (forecasting metrics) |
| `one-for-all_q04` | **Г** | Patching + embedding as TS representation for GPT-2 |
| `one-for-all_q05` | **Б** | rsLoRA stabilization, Rank-16 adapters, Parameter efficiency |
| `one-for-all_q06` | **Д** | GPT-2 frozen backbone; rsLoRA adapters in pos emb + output layers |
| `one-for-all_q07` | **Е** | Input → Patching → GPT-2 (frozen) + rsLoRA → Output Projection |
| `one-for-all_q08` | **И** | 0.55M trainable params; training configs |
| `one-for-all_q09` | **Ж** | Horizons {96–720}, 6 tasks, improvement over baselines |
| `one-for-all_q10` | **К** | Frozen GPT-2 + rsLoRA PEFT (no full fine-tuning) |
| `one-for-all_q11` | **Л** | Instance normalization, patching |

## 2. Итоговая таблица чанков

| ID | Раздел | Типы вопросов | Приоритет |
|----|--------|---------------|-----------|
| C01 | Abstract | А | Высокий |
| C02 | I. Introduction (challenges + rsLoRA) | А | Высокий |
| C03 | III. Methodology (rsLoRA + GPT-2) | А, Д, К | Высокий |
| C04 | Patching + Input Embedding | Г, Л | Средний |
| C05 | rsLoRA Design (rank stabilization) | Б, Д | Высокий |
| C06 | Training + Parameter Efficiency | И, К | Средний |
| C07 | IV. Experiments setup (datasets+metrics) | В, З | Высокий |
| C08 | IV. Results (efficiency-accuracy, horizons) | А+, Ж | Высокий |

**Всего:** 8 чанков. Переиспользуемые: C03 (А+Д+К), C05 (Б+Д).

## 3. Принципы

1. **One-for-All = rsLoRA + frozen GPT-2:** экстремально лёгкий PEFT (0.55M параметров).
2. **q02=В расширенная формулировка¹.**
