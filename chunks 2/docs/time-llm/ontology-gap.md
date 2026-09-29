# TIME-LLM: Привязка к онтологии и анализ пробелов (ревизия v1.2)

**article_id (Eval API):** `time-llm`
**Дата ревизии:** 2026-05-22
**Основание:** добавлены типы А+, В, З, Г, Б, Е, И, Ж, К, Л, М к ранее покрытым А и Д

---

## 1. Сопоставление извлечённых сущностей с классами онтологии

### Покрытые классы (обновлено для v1.2)

| Сущность | Класс онтологии | Тип вопроса | Заполняется из статьи? |
|----------|----------------|-------------|------------------------|
| TIME-LLM (reprogramming framework) | `Method` | А | Да: name="TIME-LLM", strategy="Reprogramming", prompt_type="Instruction" |
| TIME-LLM vs GPT4TS/PatchTST | `Method` (best) | А+ | Да: ablation + comparative results |
| 8 datasets (ETT, Weather, etc.) + M4 + M3 | `Dataset` | В | Да: B.2 Табл. 8 |
| MSE, MAE, SMAPE, MASE, OWA, MAPE | `Evaluation_Metric` | З | Да: B.3 |
| Patch Embeddings, Text Prototypes, Cross-Attention | `TS_Representation`, `Tokenization_Method` | Г | Да |
| Patch Reprogramming, PaP, RevIN, Lightweight params | `Improvement_Method` | Б | Да: подтверждено ablation |
| Llama-7B, GPT-2, Multi-head Cross-Attention | `BaseLLM`, `Neural_Architecture` | Д | Да |
| 5-step Pipeline | `Pipeline`, `Workflow_Step` | Е | Да |
| RevIN, Patching, Channel Independence, Trend/Lag | `DataPreprocessing` | И | Да |
| Long-term/Short-term/Few-shot/Zero-shot, horizons, improvement % | `Task_Type`, `Forecast_Horizon`, `Improvement_Percent` | Ж | Да |
| Frozen LLM, Lightweight Training, Scaling Law | `TuningStrategy`, `TrainingRegime` | К | Да |
| RevIN, Patching, Linear Embedding, Trend, Top-5 Lags | `DataPreprocessing` (normalization) | Л | Да |
| Output Projection (flatten + linear) | `Pipeline` (output step) | М | Да — уникальный тип |

### Пробелы (обновлено)

| Сущность | Проблема | Предложение | Статус после ревизии |
|----------|----------|-------------|---------------------|
| **Text Prototypes (E')** | Специфический механизм выбора V' из V через обучение W. Не сводим к обычной токенизации. | `Technique` с type="TextPrototypeSelection" или класс `TextPrototype`. | **Подтверждён** — привязан к типам Г (q04) и Б (q05) |
| **Prompt-as-Prefix (PaP)** | Три компонента промпта (dataset context, task instruction, input statistics). Уникальный prompting-паттерн. | `PromptDesign` как потенциально-расширяемый класс (3.3). | **Подтверждён** — теперь привязан к типам А (q00), Б (q05), Г (q04) |
| **Reprogramming Efficiency Metrics** | Trainable params (6.6M), GPU memory (32 GB), speed (s/iter). | `ComputationalCost` как класс 3.3. | **Подтверждён** — привязан к типам Б (q05), К (q10) |
| **QLoRA comparison** | `TuningStrategy.peft_method` требует "QLoRA". | Добавить "QLoRA" в enum. | **Подтверждён** |
| **Output Projection (тип М)** | Завершающий шаг: flatten + linear → forecasts. Уже покрыт `Pipeline` + `Workflow_Step`. | Без изменений — существующие классы покрывают. | **Покрыто** — уникальный тип М (q12) |
| **Cross-domain Zero-shot** | `Experiment.source_datasets` для zero-shot сценариев. | Использовать существующее поле. | **Покрыто** |

---

## 2. Кандидаты на расширение онтологии (обновлено)

| Кандидат | Обоснование | Приоритет | Предлагаемое решение |
|----------|-------------|-----------|---------------------|
| **TextPrototype** | Специфический механизм TIME-LLM: V' прототипов из V через W ∈ R^{V'×V}. Может появиться в TEST, S2IP-LLM. | Высокий | Новый класс `TextPrototype`: {vocab_size_V, num_prototypes_V', selection_method}. |
| **PromptDesign** | Шаблон промпта с 3 компонентами (dataset context, task instruction, input statistics). | Средний | Класс в разделе 3.3 плана. Накопить evidence из TEST, S2IP-LLM. |
| **ComputationalCost** | 6.6M params, 32 GB GPU, 0.5–0.7 s/iter — метрики эффективности. | Средний | Класс в разделе 3.3 плана. |
| **"QLoRA" в TuningStrategy.peft_method** | Статья сравнивает reprogramming с QLoRA. | Низкий | Расширить enum. |

---

## 3. Итог (обновлено)

- **Покрытие онтологии:** ~95% (13 типов вопросов, наиболее полная статья)
- **Ключевые пробелы (4):** TextPrototype, PromptDesign, ComputationalCost, QLoRA в enum
- **Снятые проблемы:** Output Projection (тип М) полностью покрыт существующими классами
- **Уникальность:** единственная статья с типом М (завершающий шаг прогноза); Output Projection как Workflow_Step
- **Рекомендация:** TextPrototype — приоритетный кандидат (подтверждён time-llm + ожидается в TEST и S2IP-LLM)
