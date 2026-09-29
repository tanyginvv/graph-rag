# TS-RAG: Извлечённые сущности (ревизия v1.2)

**article_id:** `ts-rag` | **Дата:** 2026-05-22
**Типы (v1.2, группа 2):** А (q00), В (q01), З (q02), Г (q03), Б (q04), Д (q05), Е (q06), И (q07), Ж (q08), К (q09), Л (q10), Спец (q11)

## 3.1. Сущности Eval API

### Тип А — Способы применения LLM

| Сущность | Значение | Раздел |
|----------|----------|--------|
| **TS-RAG** | RAG-based TS forecasting: retriever (Chronos encoder) → knowledge base → top-k retrieval → MoE augmentation → TSFM backbone → zero-shot forecast | Abstract, 3 |
| Lag-Llama, TimesFM, Chronos, Moirai, MOMENT, TimeGPT-1, TTMs | TSFM baselines и контекст | 2.1 |
| LLM-based: Time-LLM, GPT4TS, S2IP-LLM | LLM fine-tuning методы | 1, 2.2 |
| TaskParadigm: **Retrieval-Augmented Generation for TS** | RAG из NLP адаптирован для TS: retrieval of similar segments → augment TSFM input | 1, 3 |
| TaskParadigm: **Zero-shot via Retrieval** | Вместо fine-tuning — retrieval даёт domain adaptation без обучения на целевом датасете | 1 |

### Тип Б — Способы улучшения

| Сущность | Значение | Раздел |
|----------|----------|--------|
| **MoE Augmentation Module** | Learnable Mixture-of-Experts: динамический fusion retrieved patterns + input query embedding | 3.2 |
| **Retrieval Knowledge Base** | Chronos pretraining subset → (x_i, e_i, y_i) triplets с предвычисленными эмбеддингами | 3.1 |
| **Chronos encoder as Retriever** | Pre-trained Chronos encoder для эмбеддингов → Euclidean distance → top-k | 3.2 |
| **Adaptive Pretraining** | Обучение MoE модуля на малом subset Chronos данных (без fine-tuning TSFM backbone) | 3.1 |

### Тип В — Датасеты (7 public benchmark datasets, не специфицированы в прочитанной части)

## 3.2. Онтология

| Класс | Сущность |
|-------|----------|
| BaseLLM | Chronos (как TSFM backbone + retriever encoder) |
| TuningStrategy | Frozen TSFM backbone + trainable MoE module |
| Technique | RAG (Retrieval-Augmented Generation), MoE (Mixture-of-Experts), Euclidean Distance Retrieval, Embedding-based Similarity Search |

## 3.3. Расширяемые

| Класс | Сущность |
|-------|----------|
| Improvement_Method | RAG adaptation for TS, MoE augmentation, Knowledge Base construction |
| Baseline | Lag-Llama, TimesFM, Chronos, TimeGPT-1, Moirai, MOMENT, TTMs |
