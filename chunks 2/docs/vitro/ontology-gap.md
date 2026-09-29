# VITRO: Привязка к онтологии и анализ пробелов (ревизия v1.2)

**article_id (Eval API):** `vitro`
**Дата ревизии:** 2026-05-22
**Основание:** добавлены типы Г, Д, Е, З, Б, И, Ж, А+, К к ранее покрытым А и В

---

## 1. Сопоставление извлечённых сущностей с классами онтологии

### Покрытые классы (обновлено для v1.2)

| Сущность | Класс онтологии | Тип вопроса | Заполняется из статьи? |
|----------|----------------|-------------|------------------------|
| VITRO (основной метод) | `Method` | А | Да: name="VITRO", category="LLM-based", strategy="Inversion" ← требует enum |
| VITRO-Sim, VITRO-TimeLLM | `Method` | А, А+ | Да |
| Pseudo-word Embeddings (v_i, s) | `TS_Representation`, потенциально `PseudoWordEmbedding` | Г | Да |
| Patch Embeddings (E_i) | `TS_Representation` | Г | Да |
| GPT-2, Llama-7B | `BaseLLM`, `Neural_Architecture` | Д | Да |
| Multi-head Cross-Attention | `Neural_Architecture` | Д | Да |
| Two-Stage Pipeline (7 шагов) | `Pipeline`, `Workflow_Step` | Е | Да |
| 7 datasets (ETT, Weather, etc.) | `Dataset` | В | Да |
| MSE, MAE | `Evaluation_Metric` | З | Да |
| RevIN, Patching, Core Lexicon, Cosine Similarity | `Improvement_Method`, `Technique` | Б | Да |
| RevIN, Patching, e_stats | `DataPreprocessing` | И | Да |
| Frozen LLM, Two-stage training | `TrainingRegime`, `TuningStrategy` | И, К | Да |
| Long-term Forecasting, horizons {96,192,336,720} | `Task_Type`, `Forecast_Horizon` | Ж | Да |
| VITRO-Sim improvement 0.7–8.5% MSE | `Improvement_Percent` | Ж, А+ | Да |
| VITRO-TimeLLM improvement 0.4–4.8% MSE | `Improvement_Percent` | Ж, А+ | Да |

### Пробелы (обновлено)

| Сущность | Проблема | Предложение | Статус после ревизии |
|----------|----------|-------------|---------------------|
| **Vocabulary Inversion** | `Method.strategy` не включает "Inversion". Текущие: Alignment, Instruction, DirectProcessing, Reprogramming, Adapter, FromScratch. | Добавить `"Inversion"` в `Method.strategy`. | **Подтверждён** — теперь привязан к типам А (q00) и Г (q01) |
| **Pseudo-word Embeddings** | Нет класса для learnable embeddings, инжектируемых в LLM embedding lookup table. Ключевая техника VITRO. | Новый класс `PseudoWordEmbedding` или `Technique` с type="PseudoWordEmbedding". | **Подтверждён** — теперь явно привязан к типу Г (q01) |
| **Textual Inversion (кросс-доменная адаптация)** | VITRO адаптирует textual inversion из vision-language (text-to-image diffusion). Кросс-доменный перенос техники. | Связь `Technique → SourceDomain` (Vision-Language) или атрибут `Technique.inspired_by`. | **Подтверждён** |
| **Core Lexicon Reduction** | Промежуточное представление C = W_v V + b_v — редуцированный словарь для similarity search. | Допустимо как `Technique` с type="Projection". | **Подтверждён** — теперь привязан к типу Б (q06) |
| **Embedding-level Injection vs Fine-tuning** | VITRO модифицирует только embedding lookup table, не трогая веса LLM. Это промежуточный подход между Frozen и Fine-tuning. | Уточнить `TuningStrategy.approach`: различать "Frozen" (ничего не обучается), "EmbeddingInjection" (только embedding-уровень), "Partial" (часть слоёв). | **Новый пробел** — обнаружен при добавлении типов И (q07) и К (q10) |
| **Two-stage Pipeline (Vocabulary → Forecasting)** | Специфический двухстадийный процесс, где Stage 1 оптимизирует словарь на всём датасете, а Stage 2 применяет его. | `Pipeline` уже поддерживает `Workflow_Step`; дополнить атрибутом `pipeline_type: "two_stage_vocabulary"`. | **Новый пробел** — для типа Е (q03) |

---

## 2. Кандидаты на расширение онтологии (обновлено)

| Кандидат | Обоснование | Приоритет | Предлагаемое решение |
|----------|-------------|-----------|---------------------|
| **"Inversion" в Method.strategy** | VITRO — vocabulary inversion; принципиально отлично от Reprogramming и Alignment. Inversion = обучение новых embedding-токенов без изменения архитектуры LLM. | **Критический** | Добавить `"Inversion"` в `Method.strategy`. |
| **PseudoWordEmbedding** | Специфическая техника: обучение псевдослов как новых токенов в vocabulary LLM. Переиспользуется в TEST, S2IP-LLM. | Высокий | Новый класс или `Technique` с type="PseudoWordEmbedding". Накопить evidence из TEST, S2IP-LLM. |
| **TuningStrategy: EmbeddingInjection** | VITRO не меняет веса LLM, но модифицирует embedding lookup table. Нужно промежуточное значение между "Frozen" и "Partial". | Средний | Добавить `"EmbeddingInjection"` в `TuningStrategy.approach`. |
| **Textual Inversion (кросс-доменная)** | Адаптация техники из vision-language → TS. Полезно для отслеживания происхождения методов. | Низкий | `Technique.inspired_by` или связь с source domain. |

---

## 3. Проверка заполнения атрибутов (без изменений)

### Method (VITRO)
| Атрибут | Значение | Источник |
|---------|----------|----------|
| name | "VITRO" | Заголовок |
| category | "LLM-based" | II |
| strategy | **"Inversion"** ← требует enum | II.A (ст. 51) |
| channel_strategy | "Independent" (N variables → N univariate) | II.A (ст. 44) |
| scope | "MultiTask" (multiple datasets) | III.A |
| prompt_type | "Hybrid" (patch prompts + text prompts) | II.B (ст. 80, 112) |

### TuningStrategy (обновлено)
| Атрибут | Значение |
|---------|----------|
| approach | **"EmbeddingInjection"** ← требует расширения enum (промежуточное между Frozen и Partial) |
| peft_method | "None" |
| trainable_components | ["Pseudo-word embeddings v_i (n per dataset)", "Shared dataset embedding s", "Patch embedder W_e, b_e", "Output projection W, b", "Core lexicon projection W_v, b_v (Sim only)"] |
| frozen_components | ["GPT-2 (Sim)", "Llama-7B (Attention)"] |

---

## 4. Итог (обновлено)

- **Покрытие онтологии:** ~90% (11 типов вопросов × высокая степень покрытия; method-статья с детальным описанием)
- **Ключевые пробелы (6):**
  1. **"Inversion" в Method.strategy** (критический) — новая парадигма
  2. **PseudoWordEmbedding** (высокий) — переиспользуется в TEST, S2IP-LLM
  3. **TuningStrategy: EmbeddingInjection** (новый, средний) — промежуточный режим обучения
  4. **Textual Inversion кросс-доменная** (низкий)
  5. **Core Lexicon Reduction** (низкий) — как Technique
  6. **Pipeline: two_stage_vocabulary** (новый, низкий) — специфический тип пайплайна
- **Снятые проблемы (0):** все пробелы из v1.0 подтверждены; 2 новых обнаружено.
- **Рекомендация:** "Inversion" в `Method.strategy` — первоочередное. PseudoWordEmbedding — накопить evidence из TEST и S2IP-LLM. EmbeddingInjection — рассмотреть после полной ревизии.
