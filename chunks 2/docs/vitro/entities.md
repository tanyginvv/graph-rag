# VITRO: Извлечённые сущности (ревизия v1.2)

**article_id (Eval API):** `vitro`
**Дата ревизии:** 2026-05-22
**Типы вопросов Eval API (v1.2):** А (q00), Г (q01), Д (q02), Е (q03), В (q04), З (q05), Б (q06), И (q07), Ж (q08), А+ (q09), К (q10)
**Основание:** переход от v1.0 (2 типа: А, В) к v1.2 (11 типов); альтернативный порядок индексов

---

## 3.1. Сущности, непосредственно затребованные вопросами Eval API

### Тип А — Все способы применения LLM (`vitro_q00`)

Вопрос: *«Какие существуют способы использования LLM для прогноза и решения задач временных рядов (TS), о которых идет речь в статье «VITRO: Vocabulary Inversion for Time-series Representation Optimization»?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| LLM_Method | **VITRO** (основной метод) | Vocabulary Inversion for Time-series Representation Optimization: обучение нового датасет-специфичного словаря псевдослов (v_i для каждого TS-экземпляра + s для датасета) через textual inversion из vision-language домена; двухстадийный процесс | Abstract (ст. 20), II (ст. 40) | А |
| TaskParadigm | Vocabulary Inversion (Textual Inversion) | Адаптация textual inversion из text-to-image diffusion model literature: обучение концептов-представлений TS-данных как текстовых эмбеддингов; новая парадигма «inversion» для TS | Abstract (ст. 20), II.A (ст. 51) | А |
| LLM_Method | **VITRO-Sim** (Stage 2, вариант 1) | Similarity-based Selection: core lexicon C → cosine similarity → top-k отбор релевантных псевдослов; backbone: GPT-2 (frozen) | II.C (ст. 135–155) | А |
| LLM_Method | **VITRO-TimeLLM** (Stage 2, вариант 2) | Attention-based подход на базе TimeLLM: multi-head cross-attention между patch embeddings и оптимизированным словарём VITRO; backbone: Llama-7B (frozen) | II.C (ст. 157–167) | А |
| LLM_Method | Time-LLM (упомянут как baseline и основа для VITRO-TimeLLM) | Reprogramming TS через text prototypes + Prompt-as-Prefix; frozen LLM | I (ст. 28), III (ст. 173) | А |
| LLM_Method | TEST (упомянут в Introduction) | Text Prototype Aligned Embedding: contrastive learning для выравнивания TS и текстовых эмбеддингов | I (ст. 28) | А |
| LLM_Method | OFA / OneFitsAll (упомянут в Introduction) | Pretrained LM (GPT-2) с partial fine-tuning для TS forecasting | I (ст. 28) | А |
| LLM_Method | S2IP-LLM (упомянут в Introduction и как baseline) | Semantic Space Informed Prompt Learning: prompt learning с partial fine-tuning backbone LLM | I (ст. 28), III (ст. 173) | А |
| TaskParadigm | Frozen LLM + Pseudo-word Embeddings | Замороженная LLM (GPT-2 / Llama-7B) + инъекция специализированных псевдослов в embedding lookup table | II.B (ст. 55–68) | А |
| TaskParadigm | Patch Embeddings as Prompts | Патч-эмбеддинги E_i служат составным промптом для управления обработкой TS LLM и оптимизацией псевдослов | II.B (ст. 80) | А |

### Тип В — Бенчмарки и датасеты (`vitro_q04`)

Вопрос: *«Какие бенчмарки/датасеты используются для оценки качества прогнозирования и других задач TS, о которых идет речь в статье «VITRO: Vocabulary Inversion for Time-series Representation Optimization»?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Dataset | ETTh1 | Electricity Transformer Temperature, 1-hour; long-term forecasting | III.A (ст. 177), Табл. I | В |
| Dataset | ETTh2 | Electricity Transformer Temperature, 1-hour; long-term forecasting | III.A (ст. 177), Табл. I | В |
| Dataset | ETTm1 | Electricity Transformer Temperature, 15-min; long-term forecasting | III.A (ст. 177), Табл. I | В |
| Dataset | ETTm2 | Electricity Transformer Temperature, 15-min; long-term forecasting | III.A (ст. 177), Табл. I | В |
| Dataset | Weather | Weather data, 21 meteorological stations, 10-min; long-term forecasting | III.A (ст. 177), Табл. I | В |
| Dataset | Electricity | Electricity consumption, 321 clients, 1-hour; long-term forecasting | III.A (ст. 177), Табл. I | В |
| Dataset | Traffic | Freeway occupancy rates, 862 sensors, 1-hour; long-term forecasting | III.A (ст. 177), Табл. I | В |
| Benchmark | Long-term Forecasting (7 datasets) | 7 публичных датасетов: 4 ETT + Weather + Electricity + Traffic; стандартный бенчмарк для long-term forecasting | III.A (ст. 177) | В |
| Forecast_Horizon | {96, 192, 336, 720} | Четыре горизонта прогнозирования для long-term forecasting | III.A (ст. 177) | Ж |
| Evaluation_Metric | MSE | Mean Square Error | III.A (ст. 177), Табл. I | З |
| Evaluation_Metric | MAE | Mean Absolute Error | III.A (ст. 177), Табл. I | З |

### Тип Г — Представление временных рядов (`vitro_q01`)

Вопрос: *«Как можно использовать LLM для представления временных рядов (TS), о которых идет речь в статье...?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| TS_Representation | Pseudo-word Embeddings (v_i) | Обучение уникального псевдослова-эмбеддинга для каждого TS-экземпляра; инъекция в embedding lookup table LLM | II.B (ст. 55–68) | Г |
| TS_Representation | Shared Dataset Embedding (s) | Общий эмбеддинг доменной информации для всего датасета | II.B (ст. 59–64) | Г |
| TS_Representation | Patch Embeddings (E_i) | Линейное преобразование патчей: E_i = W_e X_{P,i} + b_e; патчи как «составной промпт» для LLM | II.B (ст. 70–80) | Г |
| Tokenization_Method | Patching | TS → P патчей длины L_p со страйдом S; P = ⌊(T-L_p)/S⌋ + 2 | II.B (ст. 70–76) | Г |
| Tokenization_Method | Vocabulary Inversion | Textual Inversion из vision-language: обучение концептов-представлений TS как текстовых эмбеддингов | II.A (ст. 51), II.B (ст. 55) | Г |

### Тип Д — Нейросетевая архитектура / backbone (`vitro_q02`)

Вопрос: *«Какую нейросетевую архитектуру используют для прогноза TS в качестве backbone...?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| BaseLLM | GPT-2 | Backbone для VITRO-Sim (Similarity-based Selection); frozen; embedding dim d | II.C (ст. 155) | Д |
| BaseLLM | Llama-7B | Backbone для VITRO-TimeLLM (Attention-based); frozen; используется с multi-head cross-attention | II.C (ст. 167) | Д |
| Neural_Architecture | Frozen decoder-only Transformer | Обе LLM (GPT-2, Llama-7B) — decoder-only; заморожены; без fine-tuning | II.B (ст. 57), II.C | Д |
| Neural_Architecture | Multi-head Cross-Attention (VITRO-TimeLLM) | Q = E_i, K/V = C (core lexicon); H голов, d_h = d/H | II.C (ст. 159–165) | Д |

### Тип Е — Пайплайн и шаги прогнозирования (`vitro_q03`)

Вопрос: *«Из каких шагов состоят пайплайны прогнозирования с использованием LLM...?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Pipeline | Two-Stage VITRO Pipeline | Stage 1 (Vocabulary Optimization) → Stage 2 (Forecasting with Learned Vocabulary) | II.A (ст. 53) | Е |
| Workflow_Step | RevIN (Reversible Instance Normalization) | Нормализация: zero mean, unit std; обратимость | II.B (ст. 70) | Е |
| Workflow_Step | Patching | Деление TS на патчи длины L_p со страйдом S | II.B (ст. 70–76) | Е |
| Workflow_Step | Linear Patch Embedding | E_i = W_e X_{P,i} + b_e → R^{P×d} | II.B (ст. 78) | Е |
| Workflow_Step | Concatenation | [E_i; v_i; s; e_stats] → вход в frozen LLM | II.B (ст. 84–88) | Е |
| Workflow_Step | Frozen LLM Processing | f([E_i; v_i; s; e_stats]) → h_i ∈ R^h | II.B (ст. 87–90) | Е |
| Workflow_Step | Output Projection | g(h_i) = W h_i + b → Ŷ_i ∈ R^τ | II.B (ст. 92–97) | Е |
| Workflow_Step | Stage 2: Similarity-based Selection | Core lexicon C → cosine sim → top-k → augmented embedding → GPT-2 | II.C (ст. 135–155) | Е |
| Workflow_Step | Stage 2: Attention-based | Multi-head cross-attention(Q=E_i, K/V=C) → Z → Llama-7B | II.C (ст. 157–167) | Е |

### Тип З — Метрики оценки качества (`vitro_q05`)

Вопрос: *«Какие метрики используются для оценки качества прогнозирования...?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Evaluation_Metric | MSE | Mean Square Error | III.A (ст. 177), Табл. I | З |
| Evaluation_Metric | MAE | Mean Absolute Error | III.A (ст. 177), Табл. I | З |

### Тип Б — Способы улучшения прогноза (`vitro_q06`)

Вопрос: *«Какие способы улучшения прогноза TS описываются в статье...?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Improvement_Method | Vocabulary Inversion | Замена general-purpose LLM vocabulary на data-centric time series vocabulary → consistent improvement | II.B (ст. 57–68), III.A (ст. 334) | Б |
| Improvement_Method | RevIN (Reversible Instance Normalization) | Смягчение distribution shift: обратимая нормализация zero-mean unit-std | II.B (ст. 70) | Б |
| Improvement_Method | Core Lexicon Reduction | C = W_v V + b_v: редукция словаря n→n' для эффективного similarity search | II.C (ст. 135–141) | Б |
| Improvement_Method | Two-stage Training | Stage 1: vocabulary optimization → Stage 2: forecasting; широкий контекст → специфичное применение | II.A (ст. 53) | Б |
| Improvement_Method | Iterative Optimization | Итеративная оптимизация псевдослов v_i и s минимизацией MSE | II.B (ст. 66, 100–129) | Б |
| Improvement_Method | Cosine Similarity Selection | Top-k отбор наиболее релевантных core lexicon embeddings для каждого патча | II.C (ст. 143–147) | Б |

### Тип И — Подготовка данных и обучение доп. параметров (`vitro_q07`)

Вопрос: *«Какие шаги подготовки данных и обучения дополнительных параметров описываются в статье...?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| DataPreprocessing | RevIN | Обратимая instance-нормализация: X̃_i = RevIN(X_i) | II.B (ст. 70) | И |
| DataPreprocessing | Patching | Сегментация TS на патчи: P = ⌊(T-L_p)/S⌋ + 2 | II.B (ст. 70–76) | И |
| DataPreprocessing | Statistical Features (e_stats) | Вычисляемые статистики X_i, добавляемые к входному представлению | II.B (ст. 84) | И |
| TrainingRegime | Two-stage Training | Stage 1: vocabulary optimization → Stage 2: forecasting | II.A (ст. 53) | И |
| TrainingRegime | Frozen LLM | Обе LLM (GPT-2, Llama-7B) полностью заморожены; без fine-tuning | II.B (ст. 55–68) | И |
| TrainingRegime | Trainable Parameters | v_i (n псевдослов), s (shared), W_e, b_e (patch embedder), W, b (output projection), W_v, b_v (core lexicon, Sim) | II.B (ст. 121–129) | И |

### Тип Ж — Тип задачи, горизонт, процентное улучшение (`vitro_q08`)

Вопрос: *«Каков тип задачи и горизонты прогнозирования, и какое процентное улучшение достигнуто...?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Task_Type | Long-term Forecasting | Многомерное долгосрочное прогнозирование; input length 512 | III.A (ст. 175–177) | Ж |
| Forecast_Horizon | {96, 192, 336, 720} | Четыре горизонта; результаты усреднены | III.A (ст. 177) | Ж |
| Improvement_Percent | VITRO-Sim vs Sim (MSE) | ↓0.7–8.5% MSE по 7 датасетам; среднее ~4.0% | III.A Табл. I | Ж |
| Improvement_Percent | VITRO-TimeLLM vs TimeLLM (MSE) | ↓0.4–4.8% MSE по 7 датасетам; среднее ~2.1% | III.A Табл. I | Ж |
| Improvement_Percent | vs S2IP-LLM (MAE) | VITRO превосходит по MAE во всех 7 датасетах | III.A (ст. 334) | Ж |
| Improvement_Percent | vs PatchTST (MAE) | VITRO лучше в 5/7 датасетов по MAE, 4/7 по MSE | III.A (ст. 334) | Ж |
| Improvement_Percent | vs DLinear | VITRO превосходит по обеим метрикам во всех датасетах | III.A (ст. 334) | Ж |

**Конкретные значения улучшений из Табл. I (VITRO-Sim vs Sim — ↓MSE):**

| Dataset | Sim MSE | VITRO-Sim MSE | Улучшение |
|---------|---------|---------------|-----------|
| ETTh1 | 0.442 | 0.412 | 6.8% |
| ETTh2 | 0.370 | 0.351 | 5.1% |
| ETTm1 | 0.365 | 0.353 | 3.3% |
| ETTm2 | 0.284 | 0.260 | 8.5% |
| Weather | 0.233 | 0.230 | 1.3% |
| Electricity | 0.165 | 0.161 | 2.4% |
| Traffic | 0.402 | 0.399 | 0.7% |

**VITRO-TimeLLM vs TimeLLM — ↓MSE:**

| Dataset | TimeLLM MSE | VITRO-TimeLLM MSE | Улучшение |
|---------|-------------|-------------------|-----------|
| ETTh1 | 0.437 | 0.416 | 4.8% |
| ETTh2 | 0.360 | 0.349 | 3.1% |
| ETTm1 | 0.367 | 0.352 | 4.1% |
| ETTm2 | 0.264 | 0.263 | 0.4% |
| Weather | 0.227 | 0.225 | 0.9% |
| Electricity | 0.168 | 0.166 | 1.2% |
| Traffic | 0.410 | 0.408 | 0.5% |

### Тип А+ — Наилучшие способы применения LLM (`vitro_q09`)

Вопрос: *«Какие наилучшие способы использования LLM для прогноза TS описываются в статье...?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| LLM_Method (best) | VITRO-Sim (наилучший по MSE) | Превосходит Sim на 0.7–8.5% MSE; лучший/второй результат в большинстве датасетов | III.A Табл. I, Results (334) | А+ |
| LLM_Method (best) | VITRO-TimeLLM (наилучший по MAE) | Превосходит TimeLLM; лучший/второй результат по MAE во многих датасетах | III.A Табл. I, Results (334) | А+ |
| LLM_Method (best) | VITRO vs S2IP-LLM | VITRO-enhanced методы превосходят S2IP-LLM по MAE во всех 7 датасетах, по MSE в 6/7 | III.A (ст. 334) | А+ |
| LLM_Method (best) | VITRO vs DLinear | VITRO превосходит DLinear по обеим метрикам во всех датасетах | III.A (ст. 334) | А+ |

### Тип К — Предобучение/дообучение LLM (`vitro_q10`)

Вопрос: *«Каковы цели и способы предобучения/дообучения LLM...?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| TuningStrategy | Frozen LLM (no fine-tuning) | GPT-2 и Llama-7B полностью заморожены; цель — сохранить pre-trained knowledge LLM | II.B (ст. 55–68) | К |
| TuningStrategy | Embedding-level Injection | Обучение только на уровне эмбеддингов: v_i, s инжектируются в embedding lookup table; архитектура LLM неизменна | II.B (ст. 57–64) | К |
| TrainingRegime | Vocabulary Optimization | Итеративная минимизация MSE для обучения псевдослов v_i и s + линейных слоёв | II.B (ст. 100–129) | К |
| TrainingRegime | No LLM Parameter Update | Обучаются только: v_i, s, W_e, b_e, W, b, W_v, b_v (Sim); параметры LLM заморожены | II.B (ст. 121–129) | К |

---

## 3.2. Дополнительные сущности (ранее «без прямых вопросов» — теперь частично покрыты типами v1.2)

| Класс | Сущность | Значение | Раздел-источник |
|-------|----------|----------|-----------------|
| BaseLLM | GPT-2 | Backbone для VITRO-Sim (Similarity-based Selection); frozen | II.C (ст. 155) |
| BaseLLM | Llama-7B | Backbone для VITRO-TimeLLM (Attention-based); frozen | II.C (ст. 167) |
| TuningStrategy | Frozen LLM | Обе LLM (GPT-2, Llama-7B) заморожены; обучаются только pseudo-word embeddings + patch embedder + output projection | II.B (ст. 55–68) |
| Technique | Textual Inversion | Метод из text-to-image diffusion model literature: обучение концепта как одного токена в text embedding space | II.A (ст. 51) |
| Technique | Vocabulary Inversion | Адаптация textual inversion для TS: обучение псевдослов v_i и s | II.B (ст. 55) |
| Technique | RevIN (Reversible Instance Normalization) | Нормализация: zero mean, unit std; обратимость для восстановления масштаба | II.B (ст. 70) |
| Technique | Patching | Деление TS на патчи длины L_p со страйдом S; P = ⌊(T-L_p)/S⌋ + 2 | II.B (ст. 70–76) |
| Technique | Linear Patch Embedding | E_i = W_e X_{P,i} + b_e | II.B (ст. 78) |
| Technique | Core Lexicon Reduction | C = h(V) = W_v V + b_v: линейная проекция для уменьшения размерности словаря | II.C (ст. 138) |
| Technique | Cosine Similarity Selection | Top-k отбор по cos(E_i, c_m) | II.C (ст. 143–147) |
| Technique | Multi-head Cross-Attention | Q = E_i, K/V = C (core lexicon); используется в VITRO-TimeLLM | II.C (ст. 159–165) |
| Technique | Prompt Templates | «The time series is [P_i^*]», «Forecast the next steps of [P_i^*]», «The dataset is [S^*]» | II.B (ст. 67), II.B (ст. 112) |
| Technique | Statistical Features (e_stats) | Вычисляемые статистики X_i, добавляемые к входному представлению | II.B (ст. 84) |
| Domain | Energy (Temperature) | ETT datasets: electric power deployment, oil temperature | III.A |
| Domain | Weather | Meteorological stations, Germany | III.A |
| Domain | Energy (Electricity) | Electricity consumption, 321 clients | III.A |
| Domain | Transportation | Freeway occupancy, California | III.A |
| ExperimentResult | MSE/MAE значения | См. Табл. I: 7 датасетов × 7 методов | III.A |

---

## 3.3. Потенциально-расширяемые классы

| Класс | Сущность | Значение | Раздел-источник |
|-------|----------|----------|-----------------|
| PromptDesign | Prompt Template | «The time series is [P_i^*], The dataset is [S^*]» — шаблон с плейсхолдерами | II.B (ст. 112) |
| PromptDesign | Patch Embeddings as Prompts | Патч-эмбеддинги как префикс промпта (аналогично Time-LLM PaP) | II.B (ст. 80) |
| TrainingRegime | Two-stage Training | Stage 1: vocabulary optimization → Stage 2: forecasting with learned vocabulary | II.A (ст. 53) |
| TrainingRegime | Frozen LLM (no fine-tuning) | LLM не дообучается; только embedding-level инъекция | II.B |
| AblationComponent | VITRO vs Standard Vocabulary | Замена general vocabulary → VITRO vocabulary: consistent improvement на всех датасетах | III.A |
| AblationComponent | VITRO-Sim vs VITRO-TimeLLM | Сравнение двух подходов Stage 2: similarity vs attention | III.A Табл. I |
| ComputationalCost | Iterative optimization overhead | VITRO's computational cost may limit application in larger datasets (упомянуто в Conclusion) | V (ст. 348) |
| CodeRepository | Time-Series-Library | `https://github.com/thuml/Time-Series-Library` — unified evaluation pipeline | III (ст. 171) |
| Baseline | Time-LLM (Jin et al. 2024) | Frozen LLM + text prototypes + PaP | III (ст. 173) |
| Baseline | S2IP-LLM (Pan et al. 2024) | Semantic Space Informed Prompt Learning (partial fine-tuning) | III (ст. 173) |
| Baseline | PatchTST (Nie et al. 2023) | Transformer with patching; best non-LLM baseline | III (ст. 173) |
| Baseline | DLinear (Zeng et al. 2023) | Simple linear model; best non-Transformer baseline | III (ст. 173) |
