# TS-RAG: Структурная разметка

**article_id (Eval API):** `ts-rag` | **Строк:** 468 | **Типы:** А (`ts-rag_q01`), Б (`ts-rag_q02`)

## Иерархия

| Раздел | Строки | Ключевое |
|--------|--------|----------|
| Abstract | 13 | RAG + MoE → zero-shot TS forecasting |
| 1. Introduction | 15–28 | Проблемы TSFM: adaptation, interpretability |
| 2. Related Work | 29–39 | TSFM (Lag-Llama, TimesFM, Chronos, Moirai, MOMENT) + RAG for TS (ReTime, TimeRAG, RAF) |
| 3. TS-RAG | 41–80+ | 3.1 Knowledge Base, 3.2 RAG+TSFM, MoE augmentation |
| 4. Experiments | далее | 7 public datasets, zero-shot SOTA |
| 5. Conclusion | конец | |

## Ключевые блоки

| Блок | Типы |
|------|------|
| Abstract (RAG + MoE + zero-shot) | А, Б |
| Introduction (TSFM limitations → RAG solution) | А |
| Related Work (TSFMs, RAG methods) | А |
| 3.1 Retrieval Knowledge Base | А, Б |
| 3.2 RAG + MoE Augmentation | А, Б |
