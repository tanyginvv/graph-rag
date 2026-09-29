# Lag-Llama: Структурная разметка

**article_id (Eval API):** `lag-llama`
**Файл:** `2310.08278v3 Lag-Llama-1.md`
**Всего строк:** 729

---

## Иерархия разделов

| Уровень | Номер | Название раздела | Строки в MD | Примечания |
|---------|-------|-----------------|-------------|------------|
| — | — | Титул + авторы + аффилиации | 1–16 | Preprint |
| 0 | — | **Abstract** | 18–20 | Аннотация |
| 1 | 1 | **Introduction** | 22–37 | Введение + вклад |
| 1 | 2 | **Related Work** | 38–45 | Обзор: statistical models → neural forecasting → foundation models |
| 1 | 3 | **Probabilistic Time Series Forecasting** | 46–82 | Формальная постановка задачи |
| 1 | 4 | **Lag-Llama** | 83–135 | Основной метод |
| 2 | 4.1 | Tokenization: Lag Features | 89–95 | Токенизация через лаговые признаки |
| 2 | 4.2 | Lag-Llama Architecture | 96–106 | Архитектура: decoder-only transformer (LLaMA-based) |
| 2 | 4.3 | Choice of Distribution Head | 108–110 | Student's t-distribution head |
| 2 | 4.4 | Value Scaling | 112–130 | Robust standardization (median + IQR) |
| 2 | 4.5 | Training Strategies | 132–134 | Stratified sampling, Freq-Mix, Freq-Mask |
| 1 | 5 | **Experimental Setup** | 136–181 | Постановка экспериментов |
| 2 | 5.1 | Datasets | 138–142 | 27 датасетов, 6 доменов, 7965 рядов, ~352M токенов |
| 2 | 5.2 | Baselines | 144–152 | 5 статистических + 7 глубоких (включая OneFitsAll на GPT-2) |
| 2 | 5.3 | Hyperparameter Search and Model Training | 174–177 | Random search 100 конфигураций, early stopping 50 эпох |
| 2 | 5.4 | Inference and Model Evaluation | 178–181 | CRPS метрика, 100 эмпирических семплов |
| 1 | 6 | **Results** | 183–219 | Результаты |
| 2 | 6.1 | Zero-Shot and Finetuning Performance on New Data | 207–214 | Табл. 1 (CRPS на 7 unseen datasets) |
| 2 | 6.2 | Few-Shot Adaptation Performance on Unseen Data | 215–219 | Табл. 2 (CRPS при 20/40/60/80% истории) |
| 1 | 7 | **Analysis** | 221–231 | Анализ |
| 2 | 7.1 | Data Diversity | 223–228 | catch22 features + PCA |
| 2 | 7.2 | Scaling Analysis | 230–231 | Neural scaling laws |
| 1 | 8 | **Discussion** | 233–237 | Обсуждение + future work |
| 1 | 9 | **Impact Statement** | 239–243 | Этические аспекты |
| 1 | 10 | **Contributions** | 245–271 | Распределение вклада авторов |
| 1 | 11 | **Acknowledgements** | 273–277 | Благодарности |
| 1 | — | **References** | 279–453 | Список литературы |
| 1 | A | **Details of Datasets** | 455–491 | Описание всех 27 датасетов |
| 1 | B | **Protocol Details** | 493–496 | Протокол обучения/валидации/тестирования |
| 1 | C | **Additional Empirical Results** | 497–501 | Результаты на pretraining datasets (Табл. 6–9) |
| 2 | C.1 | Results on the Pretraining Datasets | 499–501 | Краткое описание |
| 1 | D | **Hyperparameters of Lag-Llama** | 503–507 | Табл. 5 (гиперпараметры) |
| 1 | E | **Forecast Visualizations** | 509–529 | Описание Рис. 3–11 |
| 1 | F | **Additional Visualizations** | 531–569 | Neural Scaling Laws (Рис. 12, 13) |
| 2 | F.1 | Neural Scaling Laws | 533–565 | Формулы + параметры |

---

## Ключевые структурные блоки (для чанкинга)

| Блок | Разделы | Тип содержимого | Релевантные типы вопросов |
|------|---------|-----------------|--------------------------|
| **Аннотация** | Abstract | prose | А, Б |
| **Введение + foundation model подход** | 1. Introduction | prose | А |
| **Связанные работы (LLM-based methods)** | 2. Related Work (посл. абзац) | prose | А |
| **Формальная постановка** | 3. Probabilistic TS Forecasting | formula_heavy | А, Б (контекст) |
| **Метод: токенизация** | 4.1 Tokenization: Lag Features | prose + формулы | А, Г |
| **Метод: архитектура** | 4.2 Lag-Llama Architecture | prose | А, Д |
| **Метод: distribution head** | 4.3 Choice of Distribution Head | prose | Д |
| **Метод: масштабирование (улучшение)** | 4.4 Value Scaling | prose + формулы | Б |
| **Метод: стратегии обучения (улучшение)** | 4.5 Training Strategies | prose | Б |
| **Датасеты** | 5.1, Appendix A (Табл. 3, 4) | prose + таблицы | В |
| **Baselines** | 5.2 | prose | А, Б (контекст) |
| **Результаты Zero-shot + Finetuning** | 6.1, Табл. 1 | prose + таблица | Ж, Б |
| **Результаты Few-shot** | 6.2, Табл. 2 | prose + таблица | Ж, Б |
| **Анализ: данные + масштабирование** | 7.1, 7.2 | prose | — (3.3) |
| **Обсуждение / заключение** | 8. Discussion | prose | А, Б |
| **Конфигурации модели** | D. Table 5 | table | Д, — (3.3) |

---

## Примечания

1. **Lag-Llama — НЕ LLM-метод в чистом виде:** модель обучается с нуля (from scratch) на корпусе временных рядов, а не использует предобученную LLM. Однако в Related Work обсуждаются LLM-подходы (Time-LLM, LLM4TS, GPT2(6), OneFitsAll, UniTime, TEMPO), и OneFitsAll используется как baseline — это даёт ответ на вопрос типа А.
2. **Тип Б (улучшения):** статья содержит богатый материал: robust standardization, stratified sampling, Freq-Mix/Freq-Mask аугментации, few-shot adaptation через fine-tuning.
3. **Таблицы:** ключевые — Табл. 1 (zero-shot/finetune CRPS), Табл. 2 (few-shot CRPS), Табл. 3 (домены датасетов), Табл. 4 (статистики датасетов), Табл. 5 (гиперпараметры). Вспомогательные — Табл. 6–9 (результаты на pretraining данных).
4. **Список литературы** (строки 279–453) не подлежит чанкингу.
5. **Авторские вклады** (раздел 10, строки 245–271) и **благодарности** (раздел 11, строки 273–277) — не релевантны retrieval.
