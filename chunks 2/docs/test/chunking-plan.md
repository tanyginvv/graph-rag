# TEST: План семантического чанкинга (ревизия v1.2)

**article_id:** `test` | **Вопросов:** 12 (q00–q11), типы А–К + Спец (q11) | **Дата:** 2026-05-22

## 1. Актуальное соответствие

| question_id | Тип | Раздел-источник |
|-------------|-----|-----------------|
| `test_q00` | **А** | Abstract, 1.Introduction (TS-for-LLM paradigm), 3.Methodology |
| `test_q01` | **А+** | 4.Experiments: SOTA results across tasks |
| `test_q02` | **В** | 4.Experiments: benchmark datasets for classification/forecasting/representation |
| `test_q03` | **З** | Metrics: accuracy, F1, MSE, MAE (task-dependent) |
| `test_q04` | **Г** | Tokenization + Text-Prototype-Aligned Embedding (contrastive) |
| `test_q05` | **Б** | Contrastive alignment, Soft prompts, Frozen LLM activation |
| `test_q06` | **Д** | 8 frozen LLMs tested (GPT-2, LLaMA, BERT variants, etc.) |
| `test_q07` | **Е** | Tokenize → Encoder (CL) → Soft Prompt → Frozen LLM → Task Head |
| `test_q08` | **И** | Tokenization, patching, training configs |
| `test_q09` | **Ж** | Multiple tasks (classification, forecasting, representation), horizons, improvement % |
| `test_q10` | **К** | Frozen LLM (no fine-tuning), only encoder + soft prompts trained |
| `test_q11` | **Спец⁷** | Ablation: improvement vs baselines, generalization ability |

## 2. Итоговая таблица чанков

| ID | Раздел | Типы вопросов | Приоритет |
|----|--------|---------------|-----------|
| C01 | Abstract | А | Высокий |
| C02 | 1. Introduction (TS-for-LLM paradigm) | А | Высокий |
| C03 | 3. Methodology (TEST overview) | А, Е | Высокий |
| C04 | Tokenization + Encoder (contrastive alignment) | Г, Б | Высокий |
| C05 | Text-Prototype-Aligned Embedding | Г, Б | Высокий |
| C06 | Soft Prompts + Frozen LLM | А, Д, К | Высокий |
| C07 | 4. Experiments setup (datasets, metrics, tasks) | В, З, Ж | Высокий |
| C08 | 4. Results (classification/forecasting/representation) | А+, Ж | Высокий |
| C09 | 4. Ablation + Generalization (few-shot) | Спец, Ж | Высокий |
| C10 | 4. LLM Variants tested (8 models) | Д | Средний |

**Всего:** 10 чанков. Переиспользуемые: C03 (А+Е), C04-05 (Г+Б), C06 (А+Д+К), C07 (В+З+Ж), C08 (А++Ж).

## 3. Принципы

1. **q11=Спец:** «Демонстрирует ли подход улучшение точности и обобщающую способность» — требует ablation + generalization фрагментов.
2. **TEST = TS-for-LLM:** data-centric paradigm, контрастное выравнивание TS и текстовых эмбеддингов.
