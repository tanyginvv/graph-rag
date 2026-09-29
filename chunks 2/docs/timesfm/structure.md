# TimesFM: Структурная разметка

**article_id (Eval API):** `timesfm`
**Файл:** `2310.10688v4 A DECODER-ONLY FOUNDATION MODEL FOR TIME-SERIES Forecasting-1.md`
**Всего строк:** 561

---

## Иерархия разделов

| Уровень | Номер | Название раздела | Строки в MD | Примечания |
|---------|-------|-----------------|-------------|------------|
| — | — | Титул + авторы | 1–14 | Google Research, 4 автора |
| 0 | — | **Abstract** | 16–18 | Аннотация |
| 1 | 1 | **Introduction** | 20–34 | Введение + 2 ключевых элемента TimesFM |
| 1 | 2 | **Related Work** | 35–46 | 3 категории моделей + LLM-based методы + TimeGPT-1 |
| 1 | 3 | **Problem Definition** | 47–56 | Формальная постановка: y_{1:L} → ŷ_{L+1:L+H} |
| 1 | 4 | **Model Architecture** | 58–116 | Архитектура TimesFM |
| 2 | — | Patching | 62 | Патчинг как аналог токенов |
| 2 | — | Decoder-only model | 64 | Декодерный режим обучения |
| 2 | — | Longer output patches | 66–68 | Выходные патчи длиннее входных |
| 2 | — | Patch Masking | 70 | Маскирование для поддержки разных длин контекста |
| 2 | — | Input Layers | 74–83 | Residual Block + positional encoding |
| 2 | — | Stacked Transformer | 85–92 | n_l layers, multi-head causal self-attention + FFN |
| 2 | — | Output Layers | 94–101 | OutputResidualBlock(o_j) → ŷ |
| 2 | — | Loss Function | 103–110 | MSE point loss |
| 2 | — | Training | 112–113 | Mask sampling strategy |
| 2 | — | Inference | 114–115 | Auto-regressive decoding |
| 1 | 5 | **Pretraining Details** | 118–160 | Данные и обучение |
| 2 | — | Google Trends | 122 | 22k head queries, 15 лет, 0.5B time-points |
| 2 | — | Wiki Pageviews | 124 | ~300B time-points |
| 2 | — | Synthetic Data | 126 | 3M рядов × 2048 точек |
| 2 | — | Other real-world data | 128 | M4, Electricity, Traffic, Weather, LibCity, Favorita |
| 2 | — | Dataset Mixing and Training | 130 | Табл. 1: состав претрейнового корпуса |
| 1 | 6 | **Empirical Results** | 162–213 | Результаты |
| 2 | 6.1 | Zero-shot Evaluation | 166–186 | Monash, Darts, Informer (ETT); Figure 2 |
| 2 | 6.2 | Ablation | 188–213 | Scaling, AR decoding, input patch length, synthetic data |
| 1 | 7 | **Conclusion** | 215–217 | Заключение |
| 1 | 8 | **Impact Statement** | 219–231 | Этические аспекты |
| 1 | — | **References** | 241–329 | Список литературы |
| 1 | A | **Appendix** | 331–561 | Приложения |
| 2 | A.1 | Limitations and Future Work | 333–350 | Ограничения |
| 2 | A.2 | Metrics | 352–372 | MAE, msMAPE, scaling |
| 2 | A.3 | Finetuning Study on ETT | 374–403 | Табл. 2 (10% finetuning vs GPT4TS) |
| 2 | A.4 | Pretraining PatchTST | 405–411 | Абляция: PatchTST(ZS) |
| 2 | A.5 | Additional Empirical Results | 413–489 | Табл. 3 (Darts), Табл. 4 (Monash), Табл. 5 (ETT) |
| 2 | A.6 | More Details on Models | 491–511 | Табл. 6 (гиперпараметры TimesFM) |
| 2 | A.7 | Date Features | 513–523 | Перспективы covariates |
| 2 | A.8 | Synthetic Data | 525–534 | Детали генерации синтетических данных |
| 2 | A.9 | Illustrative Examples | 536–561 | Визуализации |

---

## Ключевые структурные блоки (для чанкинга)

| Блок | Разделы | Тип содержимого | Релевантные типы вопросов |
|------|---------|-----------------|--------------------------|
| **Аннотация** | Abstract | prose | А, Е |
| **Введение: foundation model идея** | 1. Introduction | prose | А |
| **Обзор: LLM-based методы** | 2. Related Work (посл. абзац) | prose | А |
| **Формальная постановка** | 3. Problem Definition | prose + формулы | Е (контекст) |
| **Архитектура: принципы** | 4 (Patching, Decoder-only, Output patches, Masking) | prose | Е |
| **Архитектура: компоненты** | 4 (Input Layers → Transformer → Output Layers → Loss → Training → Inference) | prose + формулы | Е, Д |
| **Претрейновые данные** | 5, Табл. 1 | prose + table | В |
| **Zero-shot результаты** | 6.1, Figure 2 | prose | Ж |
| **Абляции** | 6.2, Figure 3 | prose | — (3.3) |
| **Finetuning результаты** | A.3, Табл. 2 | table | Ж |
| **Заключение** | 7. Conclusion | prose | А |
| **Гиперпараметры** | A.6, Табл. 6 | table | Д |

---

## Примечания

1. **TimesFM — НЕ LLM-метод:** модель обучается с нуля (from scratch) на TS-корпусе (100B+ timepoints). Статья позиционирует себя как альтернативу LLM-based подходам.
2. **Главная инновация:** decoder-only training с longer output patches (128 vs 32) и patch masking для поддержки произвольных длин контекста.
3. **3 бенчмарк-группы:** Monash (18 datasets), Darts (8 datasets), Informer/ETT (4 datasets × 2 горизонта).
4. **3 размера модели:** 17M, 70M, 200M параметров.
5. **Таблицы:** Табл. 1 (состав претрейнового корпуса), Табл. 2 (finetuning), Табл. 3–5 (результаты), Табл. 6 (гиперпараметры).
6. **Figure 2 и 3** — основные визуализации результатов (scaled MAE на трёх бенчмарк-группах + абляции).
