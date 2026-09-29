# One-for-All: Структура + Сущности + Онтология + Чанкинг

**article_id (Eval API):** `one-for-all` | **Типы:** А (`one-for-all_q01`), З (`one-for-all_q02`)

## Структура

| Раздел | Строки |
|--------|--------|
| Abstract | 5 |
| I. Introduction | 17–48 |
| II. Related Work | 50–66 |
| III. Methodology (rsLoRA + patching + RevIN + GPT-2 backbone) | 70+ |
| IV. Experiments (6 tasks: long/short/few/zero-shot, classification, anomaly) | далее |

## Тип А — Способы применения LLM

| Сущность | Значение |
|----------|----------|
| **One-for-All** | rsLoRA (Gaussian Rank-Stabilized LoRA, rank=16) injected into positional embeddings + output layers; GPT-2 backbone frozen; 6 tasks unified |
| GPT4TS, Time-LLM, Voice2Series | Cross-modality adaptation baselines |
| TaskParadigm: **Parameter-Efficient Fine-tuning (PEFT)** | rsLoRA: математически обоснованная стабилизация ранга, gradient stability proof |

## Тип З — Метрики

| Метрика | Где |
|---------|-----|
| MSE, MAE | Основные метрики long-term forecasting |
| Eff.*MSE (Parameter Efficiency metric) | 5.5× better than TimesNet |
| SMAPE, MASE, OWA | Short-term forecasting (M3, M4) |

## Численные результаты

- 0.55M trainable params (6.8× меньше TimesNet, 21× меньше GPT4TS, 11.8× меньше TIME-LLM)
- 2.2 MiB memory (168–1776× меньше SOTA)
- 98.3% fewer parameters than conventional transformers
- MSE = 0.33 matches SOTA; <1% variance across 96–720 horizons

## Ontology Gap

- **rsLoRA** — специфический вариант LoRA с Gaussian rank stabilization. `TuningStrategy.peft_method = "rsLoRA"`
- **Parameter Efficiency metric (Eff.*MSE)** — новый тип метрики: произведение эффективности на точность

## Chunking Plan

| ID | Раздел | Типы | Приоритет |
|----|--------|------|-----------|
| C01 | Abstract | А, З | Высокий |
| C02 | I. Introduction + Novelties (rsLoRA + efficiency) | А | Высокий |
| C03 | III. Methodology (rsLoRA mechanism) | А | Высокий |
| C04 | IV. Experiments (metrics: MSE, MAE, SMAPE, OWA) | З | Высокий |
