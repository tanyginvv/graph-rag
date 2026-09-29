# Lag-Llama: Извлечённые сущности (ревизия v1.2)

**article_id (Eval API):** `lag-llama`
**Дата ревизии:** 2026-05-22
**Типы вопросов Eval API (v1.2):** А (q00), А+ (q01), В (q02), З (q03), Г (q04), Б (q05), Д (q06), Е (q07), И (q08), Ж (q09), К (q10), Л (q11)
**Основание:** переход от v1.0 (2 типа: А, Б с неверными индексами) к v1.2 (12 типов, стандартный порядок)

---

## 3.1. Сущности, непосредственно затребованные вопросами Eval API

### Тип А — Все способы применения LLM / больших моделей (`lag-llama_q00`)

**Важное примечание:** Lag-Llama сама по себе **не является LLM-методом** — это foundation model, обученная с нуля на временных рядах. Однако статья содержит:
1. Описание **собственного подхода** (foundation model for TS via pretraining from scratch) — что является способом применения больших моделей к прогнозированию TS
2. Обсуждение **LLM-based методов** в Related Work — как способов использования LLM для прогноза TS
3. Использование **OneFitsAll** (GPT-2 based) в качестве baseline

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| LLM_Method | Lag-Llama (foundation model approach) | General-purpose foundation model for univariate probabilistic TS forecasting; decoder-only transformer pretrained from scratch on diverse TS corpus (27 datasets, 6 domains, ~352M tokens); zero-shot + few-shot + finetuning | Abstract (ст. 20), 1 (ст. 28), 4 (ст. 83) | А |
| TaskParadigm | Pretraining from scratch | Foundation model paradigm: unsupervised pretraining on large diverse TS corpus → zero-shot/few-shot generalization to unseen datasets | 1 (ст. 26, 32–36), 4 (ст. 85) | А |
| LLM_Method | OneFitsAll (Zhou et al. 2023b) — baseline | Pretrained LLM (GPT-2) with finetuned input/output layers for TS forecasting | 2 (ст. 44), 5.2 (ст. 150) | А |
| LLM_Method | Time-LLM (Jin et al. 2023) — упомянут в Related Work | Freeze LLM encoder backbones while fine-tuning/adapting input and distribution heads | 2 (ст. 44) | А |
| LLM_Method | LLM4TS (Chang et al. 2023) — упомянут в Related Work | Two-stage fine-tuning for TS forecasting with pre-trained LLMs | 2 (ст. 44) | А |
| LLM_Method | GPT2(6) (Zhou et al. 2023a) — упомянут в Related Work | Freeze LLM encoder backbones for TS | 2 (ст. 44) | А |
| LLM_Method | UniTime (Liu et al. 2023) — упомянут в Related Work | Language-empowered unified model for cross-domain TS forecasting | 2 (ст. 44) | А |
| LLM_Method | TEMPO (Anonymous 2024) — упомянут в Related Work | Prompt-based generative pre-trained transformer for TS forecasting | 2 (ст. 44) | А |
| TaskParadigm | Freeze LLM + adapt heads | Общая парадигма LLM-for-TS: заморозить backbone LLM, обучить входные/выходные слои (адаптеры/heads) | 2 (ст. 44) | А |
| TaskParadigm | Foundation Model (from scratch) | Альтернативная парадигма: обучить модель с нуля на TS-данных, а не адаптировать LLM | 1 (ст. 26), 4 (ст. 83) | А |

### Тип Б — Способы улучшения (`lag-llama_q05`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Improvement_Method | Pretraining on diverse corpus | Обучение на 27 датасетах из 6 доменов (7965 рядов, ~352M токенов) для zero-shot/few-shot generalization | 5.1 (ст. 140), Abstract (ст. 20) | Б |
| Improvement_Method | Lag Features (tokenization) | Токенизация через лаговые признаки: x_t → k_t ∈ R^{|L|} с индексами от секунд до кварталов + F дата-временных признаков | 4.1 (ст. 89–92) | Б |
| Improvement_Method | Robust Standardization | Нормализация через median + IQR вместо mean/std для устойчивости к выбросам: x_t' = (x_t − Med) / IQR | 4.4 (ст. 114, 118–130) | Б |
| Improvement_Method | Stratified Sampling | Взвешенная выборка окон из претрейнового корпуса пропорционально числу рядов в датасете | 4.5 (ст. 134) | Б |
| Improvement_Method | Freq-Mix (frequency mixing augmentation) | Аугментация временных рядов в частотной области для снижения переобучения | 4.5 (ст. 134) | Б |
| Improvement_Method | Freq-Mask (frequency masking augmentation) | Маскирование частот при аугментации для снижения переобучения | 4.5 (ст. 134) | Б |
| Improvement_Method | Few-shot Finetuning | Дообучение pretrained модели на малых долях истории (20%–80%) новых датасетов | 6.2 (ст. 215–219) | Б |
| Improvement_Method | Student's t-distribution head | Использование t-распределения Стьюдента (3 параметра: df, mean, scale) вместо Gaussian для вероятностного прогноза | 4.3 (ст. 110) | Б |
| Improvement_Method | Rotary Positional Encoding (RoPE) | Позиционное кодирование в attention-слоях, заимствованное из LLaMA | 4.2 (ст. 100) | Б |
| Improvement_Method | RMSNorm pre-normalization | Преднормализация через RMSNorm (как в LLaMA) перед attention | 4.2 (ст. 100) | Б |
| Improvement_Method | Summary Statistics as covariates | Включение μ^i и σ^i как дополнительных ковариат (time-independent) в токены для информирования о масштабе | 4.4 (ст. 114) | Б |
| Improvement_Method | Date-time features | Включение F дата-временных признаков (second-of-minute ... quarter-of-year) для implicit frequency awareness | 4.1 (ст. 91) | Б |

### Тип В — Бенчмарки и датасеты (`lag-llama_q02`)

| Класс | Сущность | Значение | Раздел-источник |
|-------|----------|----------|-----------------|
| Dataset | Weather | D (daily), Nature domain, 30 pred length, 695 timestamps, 3010 series | Табл. 4 |
| Dataset | Pedestrian Counts | H (hourly), Transport domain, 48 pred length, 84283 timestamps, 66 series | Табл. 4 |
| Dataset | ETT-M2 | 15T (15-min), Energy domain, 24 pred length, 34560 timestamps, 1 series | Табл. 4 |
| Dataset | Platform Delay Minute | T (minute), Cloud domain, 60 pred length, 64800 timestamps, 10 series | Табл. 4 |
| Dataset | Requests Minute | T (minute), Cloud domain, 60 pred length, 64800 timestamps, 10 series | Табл. 4 |
| Dataset | Beijing PM2.5 | H (hourly), Air Quality domain, 24 pred length, 43824 timestamps, 8 series | Табл. 4 |
| Dataset | Exchange Rate | 1B (business day), Economic domain, 30 pred length, 6071 timestamps, 8 series | Табл. 4 |
| Dataset | Australian Electricity Demand | 0.5H, Energy, 60 pred length, 230676 ts, 5 series | Табл. 4 |
| Dataset | Electricity Hourly | H, Energy, 48 pred length, 26256 ts, 321 series | Табл. 4 |
| Dataset | London Smart Meters | 0.5H, Energy, 60 pred length, 23844 ts, 5560 series | Табл. 4 |
| Dataset | Solar | 10T, Energy, 60 pred length, 52500 ts, 137 series | Табл. 4 |
| Dataset | Wind Farms | T, Energy, 60 pred length, 526980 ts, 339 series | Табл. 4 |
| Dataset | Uber TLC Hourly | H, Transport, 24 pred length, 4254 ts, 262 series | Табл. 4 |
| Dataset | Traffic | H, Transport, 24 pred length, 14036 ts, 862 series | Табл. 4 |
| Dataset | KDD Cup 2018 | H, Nature, 48 pred length, 10850 ts, 270 series | Табл. 4 |
| Dataset | Sunspot | D, Nature, 30 pred length, 73894 ts, 1 series | Табл. 4 |
| Dataset | ETT H1, H2, M1 | H/15T, Energy, 24 pred length | Табл. 4 |
| Dataset | Cloud (CPU/Memory/Function/Instances) | T, Cloud, 60 pred length × 6 датасетов | Табл. 4 |
| Dataset | UCI Air Quality | H, Air Quality, 24 pred length, 9357 ts, 13 series | Табл. 4 |
| Dataset | Beijing Multisite | H, Air Quality, 24 pred length, 35064 ts, 132 series | Табл. 4 |
| Benchmark | Pretraining corpus (27 datasets) | 7965 univariate series, ~352M windows/tokens, 6 domains | 5.1 (ст. 140) |

### Тип Д — Нейросетевая архитектура / Backbone (`lag-llama_q06`)

| Класс | Сущность | Значение | Раздел-источник |
|-------|----------|----------|-----------------|
| Neural_Architecture | Decoder-only Transformer | LLaMA-based: M decoder layers, causally masked, RMSNorm + RoPE | 4.2 (ст. 98–104) |
| Backbone | Lag-Llama (custom) | 8 decoder layers, 9 heads, embedding dim 16/head, context 32; 2,449,299 params | D. Табл. 5, D (ст. 507) |
| Neural_Architecture | Linear Projection Layer | Shared linear projection: token features → hidden dimension of attention | 4.2 (ст. 100) |
| Neural_Architecture | Distribution Head (Student's t) | Projects features → (df, mean, scale) with non-linearities for positivity | 4.3 (ст. 110) |

### Тип Ж — Задача, горизонт (`lag-llama_q09`)

**Примечание:** формулировка без процентного улучшения.

| Класс | Сущность | Значение | Раздел-источник |
|-------|----------|----------|-----------------|
| Task_Type | Probabilistic Forecasting | Univariate probabilistic TS forecasting (не point forecast) | 3 (ст. 46) |
| Task_Type | Zero-shot Forecasting | На unseen datasets без дообучения | 6.1 (ст. 207) |
| Task_Type | Few-shot Adaptation | Дообучение на 20%–80% истории unseen datasets | 6.2 (ст. 215) |
| Forecast_Horizon | Различные (24, 30, 48, 60) | Prediction lengths варьируются по датасетам: 24 (ETT), 30 (Exchange Rate), 48 (Electricity, KDD), 60 (Cloud, Solar) | Табл. 4 |
| Improvement_Percent | Average Rank 2.786 (finetuned) | Lag-Llama finetuned — лучший average rank среди всех методов (разрыв 2 пункта над лучшим supervised) | 6.1 (ст. 209) |
| Improvement_Percent | Average Rank 1.429–1.857 (few-shot) | Лучший average rank на всех уровнях few-shot (20%–80%) | 6.2 Табл. 2 |

### Тип З — Метрики (`lag-llama_q03`)

| Класс | Сущность | Значение | Раздел-источник |
|-------|----------|----------|-----------------|
| Evaluation_Metric | CRPS | Continuous Ranked Probability Score — основная вероятностная метрика | 5.4 (ст. 180) |
| Evaluation_Metric | Average Rank | Средний ранг метода по всем датасетам — для оценки general-purpose качества | 5.4 (ст. 181) | З |

### Тип А+ — Наилучшие способы (`lag-llama_q01`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| LLM_Method (best) | Lag-Llama Finetuned | Avg rank 2.786 — лучший среди всех методов | 6.1 Табл. 1 | А+ |
| LLM_Method (best) | Lag-Llama Few-shot | Лучший avg rank (1.429–1.857) на всех уровнях few-shot | 6.2 Табл. 2 | А+ |

### Тип Г — Представление TS (`lag-llama_q04`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| TS_Representation | Lag Features (k_t) | x_t → k_t ∈ R^{\|L\|}: лаговые признаки из прошлых значений | 4.1 (ст. 91) | Г |
| TS_Representation | Date-time Features | F признаков для implicit frequency awareness | 4.1 (ст. 91) | Г |
| Tokenization_Method | Lag-based Tokenization | Токен размера \|L\|+F: универсально для любых частот | 4.1 (ст. 89–95) | Г |

### Тип Е — Пайплайн (`lag-llama_q07`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Pipeline | Lag-Llama Pipeline | Tokenization → Linear Proj → M Masked Decoder Layers → Distribution Head → Autoregressive Decoding | 4.2 (ст. 96–106) | Е |

### Тип И — Подготовка данных и обучение (`lag-llama_q08`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| DataPreprocessing | Robust Standardization | x_t' = (x_t − Med) / IQR; защита от выбросов | 4.4 (ст. 118–130) | И |
| DataPreprocessing | Summary Statistics | μ^i и σ^i как ковариаты в токенах | 4.4 (ст. 114) | И |
| TrainingRegime | Stratified Sampling + Augmentations | Freq-Mix/Freq-Mask; early stopping 50 эпох | 4.5 (ст. 134), 5.3 | И |

### Тип К — Предобучение/дообучение (`lag-llama_q10`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| TuningStrategy | Pretrained from Scratch | ~352M токенов, 27 датасетов; approach="FromScratch" | 5.1, 5.3 | К |
| TuningStrategy | Finetuning + Few-shot | Дообучение на downstream; 20%–80% few-shot | 6.1, 6.2 | К |

### Тип Л — Нормировка / преобразование TS (`lag-llama_q11`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| DataPreprocessing (normalization) | Robust Standardization (median + IQR) | Робастная к выбросам нормализация | 4.4 (ст. 118–130) | Л |
| DataPreprocessing (normalization) | De-standardization | Обратное преобразование sampled values × σ + μ | 4.4 (ст. 116) | Л |

---

## 3.2. Дополнительные сущности

| Класс | Сущность | Значение | Раздел-источник |
|-------|----------|----------|-----------------|
| BaseLLM | LLaMA (Touvron et al. 2023) — architectural inspiration | Архитектура Lag-Llama основана на LLaMA (decoder-only, RMSNorm, RoPE), но НЕ использует веса LLaMA | 4.2 (ст. 98) |
| BaseLLM | GPT-2 (Radford et al. 2019) — через OneFitsAll baseline | OneFitsAll использует GPT-2 как backbone LLM с дообучением heads | 5.2 (ст. 150) |
| TuningStrategy | Pretrained from scratch | Модель обучена с нуля: approach="FromScratch", нет замороженных компонентов | 4, 5.3 |
| TuningStrategy | Finetuning (downstream) | После претрейна — дообучение на конкретном датасете (fine-tuning) с early stopping 50 эпох | 5.3 (ст. 176), 6.1 |
| Technique | Lag Feature Construction | x_t → k_t ∈ R^{\|L\|}: построение лаговых признаков из прошлых значений | 4.1 (ст. 91) |
| Technique | Robust Standardization | Нормализация: median + IQR вместо mean + std | 4.4 (ст. 118–130) |
| Technique | RoPE (Rotary Positional Encoding) | Позиционное кодирование в attention (из LLaMA) | 4.2 (ст. 100) |
| Technique | RMSNorm | Root Mean Square Layer Normalization (из LLaMA) | 4.2 (ст. 100) |
| Technique | Stratified Sampling | Взвешенная выборка по числу рядов в датасете | 4.5 (ст. 134) |
| Technique | Freq-Mix / Freq-Mask | Аугментации в частотной области | 4.5 (ст. 134) |
| Technique | Student's t-distribution head | Выходной слой: 3 параметра (df, mean, scale) | 4.3 (ст. 110) |
| Technique | Greedy Autoregressive Decoding | Последовательная генерация будущих шагов | 4.2 (ст. 106) |
| Domain | Energy | Australian Electricity, Electricity Hourly, London Smart Meters, Solar, Wind Farms, ETT H1/H2/M1/M2 | Табл. 3 |
| Domain | Transport & Tourism | Traffic, Uber TLC, Pedestrian Counts | Табл. 3 |
| Domain | Nature | KDD Cup 2018, Sunspot, Weather | Табл. 3 |
| Domain | Air Quality | UCI, Beijing PM2.5, Beijing Multisite | Табл. 3 |
| Domain | Cloud | CPU/Memory/Function/Instances/Requests/Platform Delay | Табл. 3 |
| Domain | Banking & Econ | Exchange Rate | Табл. 3 |
| ExperimentResult | CRPS на 7 unseen datasets | См. Табл. 1: Weather 0.164 (zero-shot), 0.132 (finetuned); ETT-M2 0.063→0.017; и т.д. | 6.1 Табл. 1 |

---

## 3.3. Потенциально-расширяемые классы

| Класс | Сущность | Значение | Раздел-источник |
|-------|----------|----------|-----------------|
| TrainingRegime | Pretraining from scratch | Полное обучение с нуля на корпусе из 27 датасетов (~352M токенов) | 5.1, 5.3 |
| TrainingRegime | Zero-shot (no finetuning) | Оценка на unseen datasets без какого-либо дообучения | 6.1 |
| TrainingRegime | Few-shot (20%–80% data) | Дообучение на ограниченной истории unseen datasets | 6.2 |
| TrainingRegime | Full Finetuning | Дообучение на полном тренировочном наборе unseen dataset | 6.1 |
| ComputationalCost | 2,449,299 параметров | Общее число параметров финальной модели | D (ст. 507) |
| ComputationalCost | 1× NVIDIA Tesla-P100 (12 GB) | Оборудование для обучения; 4 CPU cores, 24 GB RAM | 5.3 (ст. 177) |
| ComputationalCost | Batch size 256, LR 10⁻⁴ | Параметры обучения | 5.3 (ст. 176) |
| ComputationalCost | ~352M training tokens | Размер претрейнового корпуса | 5.1 (ст. 140) |
| DataPreprocessing | Context window subsampling | Фиксированные окна размера C (32) с L дополнительными историческими точками | 3 (ст. 67), 4.1 (ст. 91) |
| DataPreprocessing | Robust Standardization | x_t' = (x_t − Med) / IQR; отдельно для каждого univariate окна | 4.4 (ст. 118–130) |
| AblationComponent | Data Diversity (catch22 + PCA) | Анализ разнообразия через 22 канонические характеристики + PCA | 7.1 (ст. 223–228) |
| AblationComponent | Scaling Laws | Neural scaling law fit: валидационная ошибка как функция от объёма данных | 7.2, F.1 (ст. 533–565) |
| CodeRepository | Hugging Face (упомянут) | Интеграция с Hugging Face для open-source release | 10 (ст. 249) |
| Baseline | DeepAR (Salinas et al. 2020) | Autoregressive RNN; strong probabilistic baseline | 5.2 (ст. 148) |
| Baseline | PatchTST (Nie et al. 2023) | Univariate Transformer with patching | 5.2 (ст. 148) |
| Baseline | TFT (Lim et al. 2021) | Temporal Fusion Transformer | 5.2 (ст. 148) |
| Baseline | N-BEATS (Oreshkin et al. 2020) | Neural basis expansion | 5.2 (ст. 150) |
| Baseline | Informer, AutoFormer, ETSFormer | Transformer-based TS forecasters | 5.2 (ст. 150) |
| Baseline | AutoARIMA, AutoETS, CrostonSBA, DynOptTheta, NPTS | Statistical baselines через AutoGluon | 5.2 (ст. 148) |
| Baseline | OneFitsAll (Zhou et al. 2023b) | Pretrained LLM (GPT-2) adapted for TS | 5.2 (ст. 150) |
