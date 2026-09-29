# TS-RAG: План семантического чанкинга (ревизия v1.2)

**article_id:** `ts-rag` | **Вопросов:** 12 (q00–q11), альтернативный порядок (группа 2) + Спец (q11) | **Дата:** 2026-05-22

## 1. Актуальное соответствие (группа 2)

| question_id | Тип | Раздел-источник |
|-------------|-----|-----------------|
| `ts-rag_q00` | **А** | Abstract, 1.Introduction, 3.TS-RAG Overview |
| `ts-rag_q01` | **В** | 4.Experiments: 7 public benchmark datasets |
| `ts-rag_q02` | **З** | MSE, MAE (standard forecasting metrics) |
| `ts-rag_q03` | **Г** | 3.1 Retrieval Knowledge Base; TS encoder for embedding |
| `ts-rag_q04` | **Б** | RAG retrieval, MoE augmentation, dynamic fusion |
| `ts-rag_q05` | **Д** | 3.3 TSFM backbone (Chronos); Retriever encoder; MoE module |
| `ts-rag_q06` | **Е** | 3: Query → Retrieve top-K → MoE Augmentation → TSFM → Forecast |
| `ts-rag_q07` | **И** | 3.1 Knowledge base construction; adaptive pretraining dataset |
| `ts-rag_q08` | **Ж** | Zero-shot forecasting; 7 datasets; horizons |
| `ts-rag_q09` | **К** | 3.4 Adaptive pretraining; frozen TSFM backbone |
| `ts-rag_q10` | **Л** | Normalization/standardization of TS inputs |
| `ts-rag_q11` | **Спец⁸** | Zero-shot improvement: up to 6.51% over TSFMs |

## 2. Итоговая таблица чанков

| ID | Раздел | Типы вопросов | Приоритет |
|----|--------|---------------|-----------|
| C01 | Abstract | А | Высокий |
| C02 | 1. Introduction (TS-RAG concept) | А | Высокий |
| C03 | 3. TS-RAG Overview (3 components) | А, Е | Высокий |
| C04 | 3.1 Retrieval Knowledge Base | Г, И | Высокий |
| C05 | 3.2 Retriever + MoE Augmentation | Г, Б, Д | Высокий |
| C06 | 3.3 TSFM Backbone (Chronos) | Д | Высокий |
| C07 | 3.4 Adaptive Pretraining + Zero-shot Inference | К, И | Средний |
| C08 | 4. Experiments setup (datasets+metrics) | В, З | Высокий |
| C09 | 4. Results (zero-shot performance, 6.51% improvement) | Спец, Ж, А+ | Высокий |

**Всего:** 9 чанков. Переиспользуемые: C03 (А+Е), C05 (Г+Б+Д), C07 (К+И), C09 (Спец+Ж+А+).

## 3. Принципы

1. **q11=Спец⁸:** «Достигает ли метод улучшения в zero-shot режиме и насколько» — требует zero-shot results фрагментов.
2. **TS-RAG = Retriever + MoE + TSFM:** retrieval-augmented generation для TS foundation models.
3. **Альтернативный порядок индексов** (группа 2): q01=В, q02=З, q03=Г, q04=Б, q05=Д.
