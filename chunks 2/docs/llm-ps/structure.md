# LLM-PS: Структурная разметка

**article_id (Eval API):** `llm-ps`
**Файл:** `LLM-PS Empowering Large Language Models for Time Series Forecasting with Temporal Patterns and Semantics.md`
**Всего строк:** 816

## Иерархия разделов

| Уровень | Номер | Название | Строки | Ключевое |
|---------|-------|----------|--------|----------|
| 0 | — | Abstract | 15–16 | MSCNN + T2T → LLM fine-tuning |
| 1 | 1 | Introduction | 19–32 | Проблема: TS patterns + semantic sparsity; вклад |
| 1 | 2 | Related Works | 33–48 | 2.1 TSF methods, 2.2 Temporal Patterns Learning |
| 1 | 3 | Approach | 49–160 | 3.1 MSCNN, 3.2 Patterns Decoupling, 3.3 T2T, 3.4 LoRA training |
| 1 | 4 | Experiments | 162–395 | Табл. 1–6 (long-term, short-term, few/zero-shot, analysis) |
| 1 | 5 | Conclusion | 396–398 | Итог |
| 1 | — | References | 408–467 | |
| 1 | A | Appendix A: Full Results | 469–663 | Табл. 7–9, A.1 Baselines, A.2 Datasets, A.3 Metrics |
| 1 | B | Appendix B: Model Analysis | 726–816 | B.1 Wavelet, B.2 Visualizations, B.3 Receptive Field, B.4 T2T details, B.5 Sensitivity |

## Ключевые блоки для чанкинга

| Блок | Разделы | Типы вопросов |
|------|---------|---------------|
| Abstract | Abstract | А, Б |
| Introduction + вклад | 1 | А |
| Related Work (LLM-based) | 2.1 (конец) | А |
| Approach overview + MSCNN | 3, 3.1 | А |
| T2T semantics extraction | 3.3 | А, Б |
| LoRA efficient training | 3.4 | Б |
| Long-term results (Табл. 1) | 4.1 | Ж |
| Short-term results (Табл. 2) | 4.2 | Ж |
| Few/Zero-shot results (Табл. 3,4) | 4.3 | Ж |
| Model Analysis (efficiency, noise) | 4.4 | Б |
| Datasets (A.2) | A.2 | В |
| Metrics (A.3) | A.3 | З |
