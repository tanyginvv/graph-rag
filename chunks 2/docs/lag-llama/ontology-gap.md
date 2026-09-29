# Lag-Llama: Привязка к онтологии и анализ пробелов (ревизия v1.2)

**article_id (Eval API):** `lag-llama`
**Дата ревизии:** 2026-05-22
**Основание:** добавлены типы А+, В, З, Г, Д, Е, И, Ж, К, Л к ранее покрытым А и Б

---

## 1. Сопоставление извлечённых сущностей с классами онтологии

### Особенности статьи

Lag-Llama — **foundation model, обученная с нуля**, не LLM-адаптация. Архитектурно вдохновлена LLaMA, но не использует веса LLM. Ключевые отличия:
- Тип А требует включения LLM-методов из Related Work наряду с собственным подходом
- Тип К: approach="FromScratch" — критический пробел онтологии (подтверждён также TimesFM)

### Покрытые классы (обновлено)

| Сущность | Класс онтологии | Тип вопроса | Заполняется? |
|----------|----------------|-------------|-------------|
| Lag-Llama (foundation model) | `Method` | А | Да: strategy="FromScratch" ← требует enum |
| Time-LLM, LLM4TS, GPT2(6), UniTime, TEMPO (Related Work) | `Method` | А | Частично (из их статей) |
| Lag-Llama Finetuned (avg rank 2.786) | `Method` (best) | А+ | Да |
| 27 datasets, 6 domains | `Dataset`, `Benchmark` | В | Да |
| CRPS, Average Rank | `Evaluation_Metric` | З | Да; CRPS — вероятностная метрика |
| Lag Features + Date-time Features | `TS_Representation`, `Tokenization_Method` | Г | Да |
| Robust Standardization, Stratified Sampling, Freq-Mix/Mask, Student's t, Few-shot finetuning | `Improvement_Method` | Б | Да |
| Decoder-only LLaMA-based, RMSNorm, RoPE, Distribution Head | `Neural_Architecture`, `Backbone` | Д | Да |
| Tokenization → Linear Proj → M Decoder Layers → Distribution Head → Autoregressive Decoding | `Pipeline`, `Workflow_Step` | Е | Да |
| Robust Standardization, Summary Statistics, Freq-Mix/Mask, Early Stopping | `DataPreprocessing`, `TrainingRegime` | И | Да |
| Univariate Probabilistic Forecasting, variable horizons | `Task_Type`, `Forecast_Horizon` | Ж | Да (без Improvement_Percent) |
| Pretrained from Scratch, Finetuning, Few-shot, Scaling Laws | `TuningStrategy`, `TrainingRegime` | К | Да |
| Robust Standardization (median+IQR), De-standardization | `DataPreprocessing` (normalization) | Л | Да |

### Пробелы (обновлено)

| Сущность | Проблема | Предложение | Статус |
|----------|----------|-------------|--------|
| **"FromScratch" в Method.strategy** | Lag-Llama обучена с нуля, не использует pre-trained LLM. Текущий enum не включает "FromScratch". | Добавить "FromScratch". | **Подтверждён** — теперь привязан к типам А (q00) и К (q10); также подтверждён TimesFM |
| **PretrainingCorpus** | ~352M токенов из 27 датасетов — значимый атрибут foundation model. | Новый класс `PretrainingCorpus`: {num_datasets, num_tokens, domains}. | **Подтверждён** |
| **FoundationModel** | Lag-Llama и TimesFM — foundation model, не LLM-адаптация. Нужен подтип Method. | `Method.model_type: "FoundationModel" | "LLM_Adaptation"`. | **Подтверждён** |
| **AverageRank** | Уникальная метрика general-purpose качества (не MSE/MAE). | `Evaluation_Metric` с name="AverageRank". | **Новый пробел** — CRPS уже есть, AverageRank — специфичен |
| **Robust Standardization** | Нормализация median+IQR вместо mean+std. | `DataPreprocessing` с method="RobustStandardization". | **Подтверждён** — привязан к типам Б, И, Л |

---

## 2. Итог (обновлено)

- **Покрытие онтологии:** ~88% (12 типов; foundation model специфика снижает полноту для А и К)
- **Ключевые пробелы (5):** FromScratch в Method.strategy, PretrainingCorpus, FoundationModel, AverageRank, Robust Standardization
- **Снятые проблемы:** Robust Standardization подтверждён как устойчивый
- **Рекомендация:** "FromScratch" — критическое расширение enum (подтверждено двумя статьями: Lag-Llama + TimesFM). PretrainingCorpus и FoundationModel — рассмотреть после ревизии TimesFM.
- **Ключевые пробелы:**
  1. `Method.strategy` не включает `"FromScratch"` — критично для foundation model статей (Lag-Llama, TimesFM).
  2. `Method.category` не включает `"FoundationModel"` — опционально, можно оставить `"Transformer"`.
  3. Нет механизма описания претрейнового корпуса (PretrainingCorpus).
  4. `Metric.name` не включает `"AverageRank"` — специфичная, но значимая метрика.
- **Рекомендация:** добавить `"FromScratch"` в `Method.strategy` немедленно (пригодится для TimesFM и Lag-Llama). Остальные расширения — накопить evidence.
