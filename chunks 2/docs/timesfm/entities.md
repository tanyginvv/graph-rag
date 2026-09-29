# TimesFM: Извлечённые сущности (ревизия v1.2)

**article_id (Eval API):** `timesfm`
**Дата ревизии:** 2026-05-22
**Типы вопросов Eval API (v1.2):** А (q00), А+ (q01), В (q02), З (q03), Г (q04), Б (q05), Д (q06), Е (q07), И (q08), Ж (q09), К (q10), Л (q11)
**Основание:** переход от v1.0 (2 типа: А, Е с неверными индексами) к v1.2 (12 типов)

---

## 3.1. Сущности, непосредственно затребованные вопросами Eval API

### Тип А — Все способы применения LLM / больших моделей (`timesfm_q00`)

Вопрос: *«Какие существуют способы использования LLM для прогноза временных рядов, о которых идет речь в статье «A Decoder-Only Foundation Model for Time Series Forecasting (TimesFM)»?»*

**Специфика:** TimesFM — foundation model, обученная с нуля (не LLM-адаптация). Статья обсуждает LLM-подходы в Related Work и позиционирует TimesFM как альтернативу.

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| LLM_Method | **TimesFM** (основной метод) | Decoder-only foundation model for TS forecasting; pretrained from scratch on 100B+ timepoints (real + synthetic); 200M params; zero-shot forecasting across domains/horizons/granularities | Abstract (ст. 16), 1 (ст. 28) | А |
| TaskParadigm | **Foundation Model from Scratch** | Обучение большой модели с нуля на массивном TS-корпусе (Google Trends, Wiki Pageviews, synthetic, M4, etc.) вместо адаптации LLM; zero-shot out-of-the-box forecasting | 1 (ст. 26–28) | А |
| LLM_Method | LLM-based fine-tuning (GPT-2) | GPT4TS / OFA: fine-tune GPT-2 backbone model for TS forecasting tasks | 2 (ст. 45) | А |
| LLM_Method | LLM as zero-shot forecaster (GPT-3, LLaMA-2) | llmtime [GFQW23]: benchmarks GPT-3 and LLaMA-2 as out-of-the-box zero-shot forecasters via prompting | 1 (ст. 33), 2 (ст. 45) | А |
| LLM_Method | TimeGPT-1 | Zero-shot foundation model for TS forecasting (closed-source, few details revealed) | 2 (ст. 45) | А |
| TaskParadigm | **Decoder-only Training** | Обучение в decoder-only режиме (аналогично GPT): модель предсказывает следующий патч на основе всех предыдущих; параллельное обучение на всём контекстном окне | 4 (ст. 64) | А |
| TaskParadigm | **Patching as Tokenization** | Разбиение TS на патчи — естественный аналог токенов в языковых моделях; улучшает производительность и скорость | 4 (ст. 62) | А |

### Тип Е — Пайплайн и шаги (`timesfm_q07`)

Вопрос: *«Из каких шагов состоит пайплайн прогнозирования в TimesFM, о которых идет речь в статье «A Decoder-Only Foundation Model for Time Series Forecasting (TimesFM)»?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Pipeline | **TimesFM Pipeline** | 1) Input Layers (patching + Residual Block + positional encoding), 2) Stacked Transformer (decoder-only, causal multi-head self-attention + FFN), 3) Output Layers (Residual Block → predictions), 4) Loss (MSE), 5) Training (mask sampling), 6) Inference (auto-regressive decoding) | 4 (ст. 58–115) | Е |
| Workflow_Step | **Patching** | Разбиение входного ряда y_{1:L} на непересекающиеся патчи размера input_patch_len (p=32) → ý_j | 4 Input Layers (ст. 74) | Е |
| Workflow_Step | **Residual Block (Input)** | MLP с одним скрытым слоем и skip-соединением: обрабатывает патч + маску в вектор размерности model_dim | 4 Input Layers (ст. 74) | Е |
| Workflow_Step | **Positional Encoding** | Стандартное синусоидальное позиционное кодирование (из оригинального Transformer) добавляется к входным токенам | 4 (ст. 83) | Е |
| Workflow_Step | **Stacked Transformer** | n_l (20 для 200M) слоёв: multi-head causal self-attention (16 голов) + feed-forward network (model_dim=1280) | 4 Stacked Transformer (ст. 85–92) | Е |
| Workflow_Step | **Causal Attention** | Каждый выходной токен может attend только к предшествующим входным токенам | 4 (ст. 85) | Е |
| Workflow_Step | **Residual Block (Output)** | Отображение выходных токенов o_j в предсказания размера output_patch_len (h=128): ý_{pj+1:pj+h} = OutputResidualBlock(o_j) | 4 Output Layers (ст. 94–101) | Е |
| Workflow_Step | **Longer Output Patches** | Выходные патчи (h=128) длиннее входных (p=32): позволяет предсказывать больший горизонт за меньшее число авторегрессионных шагов | 4 (ст. 66–68) | Е |
| Workflow_Step | **Patch Masking** | Случайное маскирование частей первого патча (r из [0, p-1]) при обучении → модель видит все возможные длины контекста от 1 до максимума | 4 Training (ст. 112) | Е |
| Workflow_Step | **Auto-regressive Decoding (Inference)** | Последовательная генерация: ŷ_{L+1:L+h}, затем конкатенация [y_{1:L}; ŷ] → ŷ_{L+h+1:L+2h}, и т.д. | 4 Inference (ст. 114) | Е |
| Workflow_Step | **MSE Loss** | Point forecasting loss: среднее MSE по всем выходным патчам в мини-батче | 4 Loss Function (ст. 103–108) | Е |
| Workflow_Step | **Standard Normalization (RevIN part)** | Контекст масштабируется на среднее и стд первого входного патча (только нормализация, без обратимости) | 5 (ст. 130) | Е |

### Тип В — Бенчмарки и датасеты (факультативно)

**Претрейновый корпус (Табл. 1):**

| Dataset | Granularity | # Series | # Time points |
|---------|-------------|----------|---------------|
| Synthetic | — | 3,000,000 | 6,144,000,000 |
| Wiki Pageviews (hourly, daily, weekly, monthly) | H/D/W/M | ~203M | ~384B |
| Google Trends (hourly, daily, weekly, monthly) | H/D/W/M | 89,740 | ~536M |
| Electricity | Hourly | 321 | 8,443,584 |
| Traffic | Hourly | 862 | 15,122,928 |
| Weather [ZZP+21] | 10 Min | 42 | 2,213,232 |
| Favorita Sales | Daily | 111,840 | 139,179,538 |
| LibCity [WJJ+23] | 15 Min | 6,159 | 34,253,622 |
| M4 (все гранулярности) | H/D/M/Q/Y | ~99,380 | ~24M |

**Бенчмарки для zero-shot оценки:**

| Benchmark | Состав | Метрика |
|-----------|--------|--------|
| **Monash** | 18 datasets (из 30, после фильтрации пропусков): finance, demand, weather, traffic; гранулярности от минут до лет | Scaled MAE (Geometric Mean) |
| **Darts** | 8 univariate datasets с сезонностями и трендами: AirPassengers, AusBeer, GasRateCO2, MonthlyMilk, Sunspots, Wine, Wooly, HeartRate | Scaled MAE |
| **Informer (ETT)** | ETTh1, ETTh2, ETTm1, ETTm2 (4 датасета); горизонты 96 и 192; контекст 512 | MAE (на нормализованных данных) |

### Тип Д — Нейросетевая архитектура (факультативно)

| Backbone | Параметры | Слои | Model dim | Heads | Input patch | Output patch |
|----------|-----------|------|-----------|-------|-------------|--------------|
| TimesFM 200M | 200M | 20 | 1280 | 16 | 32 | 128 |
| TimesFM 70M | 70M | 10 | 1024 | 16 | 32 | 128 |
| TimesFM 17M | 17M | 10 | 512 | 16 | 32 | 128 |

### Тип Ж — Результаты (факультативно)

| Класс | Сущность | Значение | Раздел-источник |
|-------|----------|----------|-----------------|
| Improvement_Percent | TimesFM vs llmtime (Monash) | >25% улучшение scaled MAE | 6.1 (ст. 176) |
| Improvement_Percent | TimesFM vs GPT4TS (finetuning) | 18% лучше на ETTh1, 12% на ETTm1 (10% data) | A.3 (ст. 378) |
| Task_Type | Zero-shot forecasting | Без какого-либо обучения на целевых датасетах | 6.1 |
| Task_Type | Few-shot finetuning | 10% обучающих данных (Appendix A.3) | A.3 |
| Forecast_Horizon | 96, 192 (ETT) | Горизонты для Informer/ETT бенчмарка | 6.1 |
| Forecast_Horizon | Variable (Monash, Darts) | Разные горизонты в зависимости от датасета | 6.1 |
| Evaluation_Metric | MAE | Основная метрика | A.2 |
| Evaluation_Metric | msMAPE | Используется в Monash для избежания undefined значений | A.2 |
| Evaluation_Metric | Scaled MAE (Geometric/Arithmetic Mean) | Нормализованная агрегация по датасетам | A.2 |

---

## 3.2. Сущности из онтологии без прямых вопросов Eval API

| Класс | Сущность | Значение | Раздел-источник |
|-------|----------|----------|-----------------|
| BaseLLM | GPT-3 (упомянут) | llmtime zero-shot baseline | 1, 2 |
| BaseLLM | LLaMA-2 (упомянут) | llmtime zero-shot baseline | 1, 2 |
| BaseLLM | GPT-2 (упомянут) | GPT4TS backbone | 2 |
| TuningStrategy | Pretrained from scratch | TimesFM: approach="FromScratch" | 5 |
| TuningStrategy | Finetuning (10% data) | Дообучение input/output residual blocks на 10% тренировочных данных | A.3 |
| TuningStrategy | Frozen + Residual Block Tuning | При finetuning: заморожен transformer stack, обучаются input/output residual blocks | A.3 |
| Technique | Patching (non-overlapping) | Разбиение на непересекающиеся патчи (в отличие от overlapping patching в PatchTST/Time-LLM) | 4 |
| Technique | Residual Block (MLP + skip) | InputResidualBlock и OutputResidualBlock | 4 |
| Technique | Causal Multi-head Self-Attention | Стандартный механизм decoder-only трансформера | 4 |
| Technique | Sinusoidal Positional Encoding | Из оригинального Transformer (Vaswani et al. 2017) | 4 |
| Technique | Patch Masking Strategy | Случайное маскирование r из [0, p-1] первого патча | 4 |
| Technique | Longer Output Patches | h=128 >> p=32 | 4 |
| Technique | Standard Normalization | Масштабирование на mean/std первого патча | 5 |
| Technique | Auto-regressive Decoding | Последовательная генерация с конкатенацией | 4 |
| Domain | Web (Google Trends, Wiki) | Специфичный домен для претрейнового корпуса | 5 |
| Domain | Energy, Transport, Weather, Finance, Demand | Традиционные домены (M4, ETT, Electricity, Traffic, Weather) | 5 |

---

## 3.3. Потенциально-расширяемые классы

| Класс | Сущность | Значение | Раздел-источник |
|-------|----------|----------|-----------------|
| TrainingRegime | Pretraining from scratch | 1.5M итераций, batch size 4096, cosine decay LR, TPUv5e-16 | 5, 6.2 |
| TrainingRegime | Zero-shot (no finetuning) | Основной режим оценки TimesFM | 6.1 |
| TrainingRegime | Few-shot finetuning (10%) | A.3 | A.3 |
| ComputationalCost | 200M параметров | Наибольшая модель; 17M и 70M также обучены | 6.2, Табл. 6 |
| ComputationalCost | TPUv5e-16, 2 дня | 1.5M итераций для 200M модели | 6.2 (ст. 194) |
| ComputationalCost | 100B+ timepoints | Общий объём претрейнового корпуса | 1 (ст. 33) |
| AblationComponent | Input patch length (8→128) | p=16,32 оптимально | 6.2 |
| AblationComponent | Output patch length (8→128) | Монотонное улучшение с ростом h | 6.2 |
| AblationComponent | Synthetic data | Ухудшение на under-represented granularities (quarterly, yearly, 10-min) без синтетики | 6.2 |
| AblationComponent | Model scaling (17M→200M) | Монотонное снижение ошибки с ростом FLOPS | 6.2 |
| DataPreprocessing | Standard Normalization | Контекст масштабируется mean/std первого патча | 5 |
| DataPreprocessing | Context window max=512 | Максимальная длина контекста (256 weekly, 64 monthly) | 5 |
| CodeRepository | — | Планируется open-weights release; код не указан явно | 8 |
| Baseline | llmtime (GPT-3/3.5) | Zero-shot LLM-forecaster | 6.1 |
| Baseline | GPT4TS | GPT-2 finetuned for TS | A.3 |
| Baseline | PatchTST | SOTA long-horizon Transformer | 6.1 |
| Baseline | N-BEATS, DeepAR, TCN, N-HiTS, ARIMA, ETS | Статистические и глубокие baseline | 6.1 |
