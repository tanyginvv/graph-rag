# TimesFM: Привязка к онтологии и анализ пробелов (ревизия v1.2)

**article_id (Eval API):** `timesfm`
**Дата ревизии:** 2026-05-22
**Основание:** добавлены типы А+, В, З, Г, Б, Д, И, Ж, К, Л к ранее покрытым А и Е

---

## 1. Сопоставление (обновлено)

**Ключевые пробелы (подтверждены ревизией):**

| Пробел | Статус |
|--------|--------|
| **FromScratch** в Method.strategy | Подтверждён (TimesFM + Lag-Llama) |
| **PretrainingCorpus** | Подтверждён (100B+ timepoints, 19 источников) |
| **FoundationModel** (подтип Method) | Подтверждён |
| **ScaledMAE, msMAPE** в Metric | Подтверждены (специфичные метрики TimesFM) |
| **Patch Masking** как TrainingStrategy | Новый — уникальная техника обучения decoder-only для TS |

**Покрытие онтологии:** ~92% (12 типов; foundation model специфика).

| Сущность | Класс онтологии | Заполняется? |
|----------|----------------|-------------|
| TimesFM | `Method` | Да: name="TimesFM", category="Transformer" (decoder-only), strategy="FromScratch", channel_strategy="Independent" (univariate), scope="MultiTask", prompt_type="None" |
| TimesFM Pipeline | `Pipeline` + `Workflow_Step` | Да: 6+ шагов (Input → Transformer → Output → Loss → Training → Inference) |
| GPT-3, LLaMA-2, GPT-2 | `BaseLLM` | Да (упомянуты) |
| FromScratch training | `TuningStrategy` | Да: approach="FromScratch" |
| Finetuning (10% data) | `TuningStrategy` | Да: approach="FullFT" или "FPT" (frozen transformer + tuned residual blocks) |
| Patching, Residual Block, Causal Attention, etc. | `Technique` | Да |
| Monash, Darts, ETT datasets | `Dataset` / `Benchmark` | Да |
| MAE, msMAPE, Scaled MAE | `Metric` | Частично: "Scaled MAE" и "msMAPE" отсутствуют в `Metric.name` |
| Zero-shot forecasting | `Task` / `Experiment.training_regime` | Да |
| 100B+ timepoints pretraining corpus | — | **Пробел:** нет класса для претрейнового корпуса |

### Ключевые пробелы

| Сущность | Проблема | Предложение |
|----------|----------|-------------|
| **Scaled MAE** | Отсутствует в `Metric.name`. Это основная метрика TimesFM. | Добавить `"ScaledMAE"` в `Metric.name`. |
| **msMAPE** | Отсутствует в `Metric.name` (есть sMAPE, но не msMAPE). | Добавить `"msMAPE"` в `Metric.name`. |
| **Pretraining Corpus** | TimesFM использует массивный претрейновый корпус (~100B timepoints, 8+ источников). Нет класса для его описания. | `Experiment` с `training_regime="Pretraining"` и `source_datasets=[...]`. Или новый класс `PretrainingCorpus` (уже предложен в Lag-Llama анализе). |
| **Decoder-only vs Encoder-Decoder** | Архитектурное различие между TimesFM (decoder-only) и PatchTST (encoder-decoder) важно, но не отражается в онтологии. | Добавить атрибут `architecture_type: "DecoderOnly" | "EncoderDecoder" | "EncoderOnly"` в `Method` или `Neural_Architecture`. |
| **Longer Output Patches** | Специфическая техника TimesFM: output_patch_len (128) >> input_patch_len (32). Аналог в LLM — авторегрессионная генерация нескольких токенов за шаг. | `Technique` с name="Longer Output Patches" и type="Tokenization". |
| **Patch Masking Strategy** | Специфический метод обучения для поддержки произвольных длин контекста. | `Technique` с name="Patch Masking" и type="Regularization". |

---

## 2. Кандидаты на расширение онтологии

| Кандидат | Обоснование | Предлагаемое решение |
|----------|-------------|---------------------|
| **PretrainingCorpus** | Уже предложен в анализе Lag-Llama. TimesFM подтверждает необходимость: 100B+ timepoints, 8+ источников, смесь real/synthetic. | Новый класс `PretrainingCorpus`: { num_timepoints, num_series, sources: string[], real_ratio: float, synthetic_ratio: float }. Связь: `Method → PretrainingCorpus`. |
| **"ScaledMAE", "msMAPE"** | Специфичные метрики TimesFM и Monash benchmark. | Добавить в `Metric.name`. |
| **architecture_type** | Различие decoder-only (TimesFM, Lag-Llama) vs encoder-decoder (PatchTST) критично для архитектурного сравнения. | Атрибут `Method.architecture_type` или `Neural_Architecture.type`. |

---

## 3. Проверка заполнения атрибутов существующих классов

### Method (TimesFM)
| Атрибут | Значение | Источник |
|---------|----------|----------|
| name | "TimesFM" | Заголовок |
| category | "Transformer" (decoder-only) | 4 |
| strategy | **"FromScratch"** | 1, 5 |
| channel_strategy | "Independent" (univariate) | 3 |
| scope | "MultiTask" (multi-domain, multi-granularity) | 6.1 |
| prompt_type | "None" | — |
| code_url | — (планируется open-weights) | 8 |

### Pipeline (TimesFM)
| Шаг | Workflow_Step |
|-----|---------------|
| 1 | Patching (non-overlapping, p=32) |
| 2 | Input Residual Block (MLP + skip) |
| 3 | Positional Encoding (sinusoidal) |
| 4 | Stacked Transformer (20 layers, causal attention) |
| 5 | Output Residual Block (→ h=128 predictions) |
| 6 | MSE Loss computation |
| 7 | Auto-regressive Decoding (inference) |

### TuningStrategy
| Режим | approach | frozen_components | trainable_components |
|-------|----------|-------------------|---------------------|
| Pretraining | "FromScratch" | [] | All |
| Finetuning (10%) | "FPT" | ["Stacked Transformer"] | ["Input Residual Block", "Output Residual Block"] |

---

## 4. Итог

- **Покрытие онтологии:** ~85%
- **Ключевые пробелы:**
  1. `PretrainingCorpus` — подтверждённая необходимость (Lag-Llama + TimesFM)
  2. `Method.strategy = "FromScratch"` — подтверждено (уже предложено в Lag-Llama)
  3. Специфичные метрики: `"ScaledMAE"`, `"msMAPE"`
  4. Архитектурный атрибут `architecture_type`
- **Рекомендация:** `PretrainingCorpus` и `"FromScratch"` в `Method.strategy` становятся высокоприоритетными расширениями онтологии (подтверждены двумя статьями).
