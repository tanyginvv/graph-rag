# TIME-LLM: Извлечённые сущности (ревизия v1.2)

**article_id (Eval API):** `time-llm`
**Дата ревизии:** 2026-05-22
**Типы вопросов Eval API (v1.2):** А (q00), А+ (q01), В (q02), З (q03), Г (q04), Б (q05), Д (q06), Е (q07), И (q08), Ж (q09), К (q10), Л (q11), М (q12)
**Основание:** переход от v1.0 (2 типа: А, Д с неверными индексами) к v1.2 (13 типов, стандартный порядок)

---

## 3.1. Сущности, непосредственно затребованные вопросами Eval API

### Тип А — Все способы применения LLM (`time-llm_q00`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| LLM_Method | TIME-LLM | Reprogramming framework: repurpose frozen LLM for TS forecasting via text prototypes + Prompt-as-Prefix | Abstract (ст. 26), 1. Introduction (ст. 44), 3. Methodology (ст. 64) | А |
| TaskParadigm | Reprogramming | Input TS → text prototype representation → frozen LLM → output projection; backbone LLM kept intact, no fine-tuning | Abstract (ст. 26), 1 (ст. 44–48), 3 (ст. 64–72) | А |
| TaskParadigm | Cross-modality Adaptation | Transfer knowledge from pre-trained NLP/CV models to TS via reprogramming (not fine-tuning, not direct input editing) | 2. Related Work (ст. 62), A (ст. 490) | А |
| LLM_Method | Prompt-as-Prefix (PaP) | Prompts as prefixes to enrich input context + guide transformation of reprogrammed patches; components: dataset context, task instruction, input statistics | Abstract (ст. 26), 3.1 Prompt-as-Prefix (ст. 101–108) | А |
| LLM_Method | Patch Reprogramming | Reprogram input patches using pre-trained word embeddings E → small collection of text prototypes E' via multi-head cross-attention | 3.1 Patch Reprogramming (ст. 86–98) | А |
| TaskParadigm | Frozen LLM | Backbone LLM kept frozen; only lightweight input transformation + output projection trained | 3 (ст. 72), 4.5 (ст. 348) | А |

### Тип Д — Нейросетевая архитектура / Backbone (`time-llm_q06`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Neural_Architecture | Transformer (Decoder-only) | Llama-7B (32 layers), GPT-2 (12 layers), GPT-2 (6 layers) — все decoder-only Transformer-based LLM | 3 (ст. 66), 4.5 (ст. 326–327), B.1 (ст. 496) | Д |
| Backbone | Llama-7B (default) | Llama-7B, 32 Transformer layers, full capacity; open-source by Touvron et al. 2023 | 4 (ст. 113), B.1 (ст. 496) | Д |
| Backbone | Llama (8 layers) | Llama-7B at 1/4 capacity: first 8 Transformer layers | 4.5 (ст. 326), F. Табл. 17 | Д |
| Backbone | GPT-2 (12) | GPT-2 with 12 Transformer layers | 4.5 (ст. 327), F. Табл. 17 | Д |
| Backbone | GPT-2 (6) | GPT-2 with 6 Transformer layers | 4.5 (ст. 327), F. Табл. 17 | Д |
| Neural_Architecture | Multi-head Cross-Attention | K-head cross-attention for patch reprogramming: Q from patches, K/V from text prototypes | 3.1 (ст. 90–98) | Д |
| Neural_Architecture | Linear Patch Embedder | Simple linear layer to embed patches into dimension d_m | 3.1 Input Embedding (ст. 85) | Д |
| Neural_Architecture | Linear Output Projection | Flatten + linear projection of LLM output representations to final forecasts | 3.1 Output Projection (ст. 109) | Д |

### Тип В — Бенчмарки и датасеты (`time-llm_q02`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Dataset | ETTh1 | Electricity Transformer Temperature, 1-hour, 7 variates, (8545, 2881, 2881) | B.2 Табл. 8 | В |
| Dataset | ETTh2 | Electricity Transformer Temperature, 1-hour, 7 variates, (8545, 2881, 2881) | B.2 Табл. 8 | В |
| Dataset | ETTm1 | Electricity Transformer Temperature, 15-min, 7 variates, (34465, 11521, 11521) | B.2 Табл. 8 | В |
| Dataset | ETTm2 | Electricity Transformer Temperature, 15-min, 7 variates, (34465, 11521, 11521) | B.2 Табл. 8 | В |
| Dataset | Weather | Weather, 10-min, 21 variates, (36792, 5271, 10540) | B.2 Табл. 8 | В |
| Dataset | Electricity (ECL) | Electricity consumption, 1-hour, 321 variates, (18317, 2633, 5261) | B.2 Табл. 8 | В |
| Dataset | Traffic | Freeway occupancy rates, 1-hour, 862 variates, (12185, 1757, 3509) | B.2 Табл. 8 | В |
| Dataset | ILI | Influenza-like illness, 1-week, 7 variates, (617, 74, 170) | B.2 Табл. 8 | В |
| Benchmark | M4 | 100K time series, 6 frequencies (Yearly–Hourly), business/financial/economic | 4.2 (ст. 172), B.2 Табл. 8 | В |
| Dataset | M3-Quarterly | 756 quarterly series, 5 domains, forecasting horizon 8 | B.2 Табл. 8, D.2 | В |
| Dataset | M4-Yearly | 23000 yearly series, Demographic domain | B.2 Табл. 8 | В |
| Dataset | M4-Quarterly | 24000 quarterly series, Finance domain | B.2 Табл. 8 | В |
| Dataset | M4-Monthly | 48000 monthly series, Industry domain | B.2 Табл. 8 | В |
| Dataset | M4-Weekly | 359 weekly series, Macro domain | B.2 Табл. 8 | В |
| Dataset | M4-Daily | 4227 daily series, Micro domain | B.2 Табл. 8 | В |
| Dataset | M4-Hourly | 414 hourly series, Other domain | B.2 Табл. 8 | В |
| Benchmark | ETT | Electricity Transformer Temperature — 4 под-датасета (ETTh1, ETTh2, ETTm1, ETTm2) | B.2 (ст. 524) | В |

### Тип Ж — Тип задачи, горизонт, улучшение (`time-llm_q09`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Task_Type | Long-term Forecasting | Multivariate time series forecasting, horizon H ∈ {96, 192, 336, 720} (ILI: {24, 36, 48, 60}) | 4.1 (ст. 119) | Ж |
| Task_Type | Short-term Forecasting | Univariate time series forecasting, horizon H ∈ [6, 48] | 4.2 (ст. 172) | Ж |
| Task_Type | Few-shot Forecasting | 10% and 5% training data scenarios | 4.3 (ст. 186) | Ж |
| Task_Type | Zero-shot Forecasting | Cross-domain adaptation: train on source dataset, test on target dataset | 4.4 (ст. 286) | Ж |
| Forecast_Horizon | H=96 | Long-term horizon | 4.1 (ст. 119) | Ж |
| Forecast_Horizon | H=192 | Long-term horizon | 4.1 (ст. 119) | Ж |
| Forecast_Horizon | H=336 | Long-term horizon | 4.1 (ст. 119) | Ж |
| Forecast_Horizon | H=720 | Long-term horizon | 4.1 (ст. 119) | Ж |
| Forecast_Horizon | H ∈ [6, 48] | Short-term horizons (M4) | 4.2 (ст. 172) | Ж |
| Improvement_Percent | 12% over GPT4TS | Average MSE reduction in long-term forecasting | 4.1 (ст. 122) | Ж |
| Improvement_Percent | 20% over TimesNet | Average MSE reduction in long-term forecasting | 4.1 (ст. 122) | Ж |
| Improvement_Percent | 1.4% over PatchTST | Average MSE reduction in long-term forecasting | 4.1 (ст. 122) | Ж |
| Improvement_Percent | 8.7% over GPT4TS | Short-term (overall SMAPE/MASE/OWA) | 4.2 (ст. 174) | Ж |
| Improvement_Percent | 5% over GPT4TS | 10% few-shot MSE reduction | 4.3 (ст. 190) | Ж |
| Improvement_Percent | 14.2% over second-best | Zero-shot MSE reduction | 4.4 (ст. 288) | Ж |
| Improvement_Percent | 75% over LLMTime | Zero-shot improvement vs LLMTime (7B) | 4.4 (ст. 288) | Ж |

### Тип З — Метрики оценки (`time-llm_q03`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Evaluation_Metric | MSE | Mean Square Error — основная метрика для long-term forecasting | B.3 (ст. 533) | З |
| Evaluation_Metric | MAE | Mean Absolute Error — вторая метрика для long-term forecasting | B.3 (ст. 537) | З |
| Evaluation_Metric | SMAPE | Symmetric Mean Absolute Percentage Error — для M4 short-term | B.3 (ст. 541) | З |
| Evaluation_Metric | MASE | Mean Absolute Scaled Error — для M4 short-term | B.3 (ст. 549) | З |
| Evaluation_Metric | OWA | Overall Weighted Average — специфичная метрика M4 competition | B.3 (ст. 553) | З |
| Evaluation_Metric | MAPE | Mean Absolute Percentage Error — дополнительная (M3-Quarterly) | B.3 (ст. 545), D.2 Табл. 13 | З |
| Evaluation_Metric | MRAE | Mean Relative Absolute Error — дополнительная (M3-Quarterly) | D.2 Табл. 13 | З |

### Тип Е — Пайплайн (`time-llm_q07`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Pipeline | TIME-LLM Pipeline | 1) Input Embedding (RevIN + patching + linear embed), 2) Patch Reprogramming (multi-head cross-attention with text prototypes), 3) Prompt-as-Prefix (dataset context + task instruction + input statistics), 4) Frozen LLM feedforward, 5) Output Projection (flatten + linear) | 3 (ст. 72), Рис. 2 | Е |
| Workflow_Step | RevIN | Reversible instance normalization for distribution shift mitigation | 3.1 Input Embedding (ст. 78) | Е |
| Workflow_Step | Patching | Segment TS into overlapped/non-overlapped patches of length Lp with stride S | 3.1 Input Embedding (ст. 78–84) | Е |
| Workflow_Step | Text Prototype Learning | Learn W ∈ R^{V'×V} to select V' prototypes from V vocabulary embeddings | B.1 (ст. 498) | Е |
| Workflow_Step | Output Flatten + Project | Flatten P×D output to 1D, then linear projection to H forecasts | B.1 (ст. 498) | Е |

### Тип А+ — Наилучшие способы применения LLM (`time-llm_q01`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| LLM_Method (best) | TIME-LLM (full Llama-7B) | Наилучший backbone: Llama-7B (32 layers) превосходит GPT-2 на 14.7% MSE | 4.5 (ст. 326) | А+ |
| LLM_Method (best) | TIME-LLM (Patch Reprogramming + PaP) | Абляция подтверждает: удаление Patch Reprogramming → +9.2% MSE; удаление PaP → +8% MSE | 4.5 (ст. 328) | А+ |
| LLM_Method (best) | TIME-LLM vs GPT4TS (long-term) | 12% среднее снижение MSE; превосходит по большинству датасетов | 4.1 (ст. 121–122) | А+ |
| LLM_Method (best) | TIME-LLM vs PatchTST | 1.4% среднее снижение MSE (SOTA task-specific Transformer) | 4.1 (ст. 122) | А+ |
| LLM_Method (best) | TIME-LLM zero-shot | 14.2% over second-best; 75% over LLMTime (7B) | 4.4 (ст. 288) | А+ |

### Тип Г — Представление временных рядов (`time-llm_q04`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| TS_Representation | Patch Embeddings (X̂_P) | TS → P патчей → линейный embedder → R^{P×d_m} | 3.1 Input Embedding (ст. 78–85) | Г |
| TS_Representation | Text Prototypes (E') | Линейный probe E ∈ R^{V×D} → E' ∈ R^{V'×D}, V' ⊆ V; «short up», «steady down» как языковые cues | 3.1 Patch Reprogramming (ст. 88) | Г |
| Tokenization_Method | Patching | TS → P патчей длины L_p со страйдом S; P = ⌊(T-L_p)/S⌋ + 2 | 3.1 Input Embedding (ст. 78–84) | Г |
| Tokenization_Method | Reprogramming via Cross-Attention | Q = patches, K/V = text prototypes E'; multi-head cross-attention → O^{(i)} ∈ R^{P×D} | 3.1 Patch Reprogramming (ст. 90–98) | Г |

### Тип Б — Способы улучшения прогноза (`time-llm_q05`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Improvement_Method | Patch Reprogramming | Выравнивание модальностей TS и NLP через text prototypes; абляция: +9.2% MSE без него | 3.1 (ст. 86–98), 4.5 (ст. 328) | Б |
| Improvement_Method | Prompt-as-Prefix (PaP) | 3 компонента промпта обогащают входной контекст; абляция: +8% MSE без него | 3.1 (ст. 101–108), 4.5 (ст. 328) | Б |
| Improvement_Method | Statistical Context in PaP | Тренды + top-5 лагов; наибольший вклад среди компонент PaP: +10.2% MSE без них | 4.5 (ст. 328) | Б |
| Improvement_Method | RevIN | Обратимая instance-нормализация для смягчения distribution shift | 3.1 (ст. 78) | Б |
| Improvement_Method | Lightweight Reprogramming | 6.6M обучаемых параметров (0.2% от Llama-7B); эффективнее QLoRA | 4.5 (ст. 348) | Б |

### Тип И — Подготовка данных и обучение доп. параметров (`time-llm_q08`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| DataPreprocessing | RevIN | X̃^{(i)} = RevIN(X^{(i)}): zero mean, unit std; обратимость | 3.1 (ст. 78) | И |
| DataPreprocessing | Patching | Сегментация TS на патчи длины L_p со страйдом S | 3.1 (ст. 78–84) | И |
| DataPreprocessing | Channel Independence | N многомерных переменных → N одномерных рядов, обрабатываются независимо | 3 (ст. 72) | И |
| TrainingRegime | Frozen LLM | Backbone LLM заморожен; обучаются только Input Embedding, Patch Reprogramming, Output Projection | 3 (ст. 72) | И |
| TrainingRegime | Standard Training | Полные обучающие наборы; 50–100 эпох; Adam optimizer, LR 10⁻²–10⁻⁴ | B.4 Табл. 9 | И |
| DataPreprocessing | Trend + Lag Calculation | Сумма разностей для тренда; FFT-автокорреляция для top-5 лагов | B.1 (ст. 498) | И |

### Тип К — Предобучение/дообучение LLM (`time-llm_q10`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| TuningStrategy | Frozen (no fine-tuning) | Backbone LLM полностью заморожен; цель — сохранить pre-trained knowledge | 3 (ст. 72) | К |
| TuningStrategy | Reprogramming (vs Fine-tuning) | Не fine-tuning, а reprogramming: адаптация через input transformation + output projection | 1 (ст. 44), 3 (ст. 72) | К |
| TrainingRegime | Lightweight Training | 6.6M обучаемых параметров (0.2% от 3405M); сравнение с QLoRA | 4.5 (ст. 348) | К |
| TrainingRegime | Scaling Law Preserved | Llama (32) > Llama (8) на 14.5%; Llama > GPT-2 на 14.7% | 4.5 (ст. 326–327) | К |

### Тип Л — Нормировка / преобразование TS (`time-llm_q11`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| DataPreprocessing (normalization) | RevIN | Обратимая instance-нормализация: zero mean, unit std; обратимость для восстановления масштаба | 3.1 (ст. 78) | Л |
| DataPreprocessing (normalization) | Patching | TS → перекрывающиеся/неперекрывающиеся патчи; семантическая агрегация + сжатие | 3.1 (ст. 78–84) | Л |
| DataPreprocessing (normalization) | Linear Patch Embedding | Линейное преобразование патчей в размерность d_m | 3.1 (ст. 84–85) | Л |
| DataPreprocessing (normalization) | Trend Calculation | Сумма разностей последовательных шагов для определения тренда (↑/↓) | B.1 (ст. 498) | Л |
| DataPreprocessing (normalization) | Top-5 Lag Detection | FFT-автокорреляция → выбор 5 лагов с наибольшей корреляцией | B.1 (ст. 498) | Л |

### Тип М — Завершающий шаг прогноза (`time-llm_q12`)

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Pipeline (output) | Output Projection | Flatten P×D → 1D (длина P×D) → линейная проекция → Ŷ^{(i)} ∈ R^H | 3.1 (ст. 109), B.1 (ст. 498) | М |
| Pipeline (output) | Discard Prefix | Отбрасывание префиксной части (промпт) после feedforward через frozen LLM | B.1 (ст. 498) | М |

---

## 3.2. Дополнительные сущности (ранее «без прямых вопросов» — теперь частично покрыты типами v1.2)

| Класс | Сущность | Значение | Раздел-источник |
|-------|----------|----------|-----------------|
| BaseLLM | Llama-7B | Llama by Touvron et al. 2023, 7B parameters, open-source | 4 (ст. 113), B.1 |
| BaseLLM | GPT-2 | GPT-2 by Radford et al. 2019, open-source | 3 (ст. 66) |
| TuningStrategy | Frozen | Backbone LLM полностью заморожен; обучаются только Input Embedding, Patch Reprogramming, Output Projection | 3 (ст. 72), 4.5 (ст. 348) |
| TuningStrategy | QLoRA (comparison) | Параметро-эффективный fine-tuning для сравнения эффективности | 4.5 (ст. 348), G (ст. 876) |
| Technique | Instance Normalization (RevIN) | Обратимая instance-нормализация: zero mean, unit std; обратимость для восстановления масштаба | 3.1 Input Embedding (ст. 78) |
| Technique | Patching | Деление TS на патчи длины Lp со страйдом S; P = ⌊(T-Lp)/S⌋ + 2 | 3.1 Input Embedding (ст. 78–84) |
| Technique | Multi-head Cross-Attention | K-head cross-attention: Q=patches, K/V=text prototypes; d = ⌊dm/K⌋ | 3.1 Patch Reprogramming (ст. 90–98) |
| Technique | Prompt-as-Prefix (PaP) | Три компонента промпта: dataset context, task instruction, input statistics (trends + top-5 lags) | 3.1 Prompt-as-Prefix (ст. 101–108) |
| Technique | Text Prototype Selection | Обучение W ∈ R^{V'×V} для выбора V' прототипов из V словарных встраиваний | B.1 (ст. 498) |
| Technique | Trend Calculation | Сумма разностей последовательных шагов: >0 = восходящий, <0 = нисходящий | B.1 (ст. 498) |
| Technique | Top-5 Lag Detection | Автокорреляция через FFT → выбор 5 лагов с наибольшей корреляцией | B.1 (ст. 498) |
| Domain | Temperature (Energy) | ETT датасеты — electric power deployment, oil temperature | B.2 (ст. 524) |
| Domain | Electricity (Energy) | Electricity consumption, 321 customers | B.2 Табл. 8 |
| Domain | Weather | Meteorological stations, Germany | B.2 Табл. 8 |
| Domain | Transportation | Freeway occupancy, California | B.2 Табл. 8 |
| Domain | Health | Influenza-like illness, CDC | B.2 Табл. 8 |
| Domain | Mixed (M4) | Business, financial, economic forecasting | B.2 (ст. 526) |
| ExperimentResult | См. Табл. 1–5, 10–16 | Множественные результаты MSE/MAE на всех датасетах и горизонтах | 4.1–4.4, D–E |

---

## 3.3. Потенциально-расширяемые классы (для будущих вопросов)

| Класс | Сущность | Значение | Раздел-источник |
|-------|----------|----------|-----------------|
| PromptDesign | Prompt Template (Рис. 4) | Placeholder-based шаблон: `<>` для dataset context, task instruction, input statistics | 3.1 Prompt-as-Prefix (ст. 105–107), Рис. 4 |
| TrainingRegime | Standard (full data) | Полные обучающие наборы для long-term/short-term | 4.1, 4.2 |
| TrainingRegime | Few-shot (10%) | 10% обучающих данных | 4.3 |
| TrainingRegime | Few-shot (5%) | 5% обучающих данных | 4.3 |
| TrainingRegime | Zero-shot (cross-domain) | Обучение на source dataset, тест на target dataset (ETT cross-domain) | 4.4 |
| ComputationalCost | 6.6M trainable params | Только 0.2% от общего числа параметров Llama-7B (3405M) | 4.5 (ст. 348) |
| ComputationalCost | 32 GB GPU memory | Llama full capacity on A100-80G | 4.5 Табл. 7 |
| ComputationalCost | Training speed 0.5–0.7 s/iter | Llama full capacity | 4.5 Табл. 7 |
| ComputationalCost | 71.2% param reduction vs QLoRA | Reprogramming vs PEFT efficiency | G (ст. 878) |
| AblationComponent | w/o Patch Reprogramming | MSE degradation 9.2% (standard), >17% (few-shot) | 4.5 (ст. 328) |
| AblationComponent | w/o Prompt-as-Prefix | MSE degradation >8% (standard), >19% (few-shot) | 4.5 (ст. 328) |
| AblationComponent | w/o Statistical Context | MSE increase 10.2% — наибольшая деградация среди компонентов промпта | 4.5 (ст. 328) |
| AblationComponent | w/o Dataset Context | MSE increase >9.6% | 4.5 Табл. 6 |
| AblationComponent | w/o Task Instruction | MSE increase >7.7% | 4.5 Табл. 6 |
| AblationComponent | Llama (8) vs Llama (32) | Scaling law preserved: 14.5% degradation with 1/4 capacity | 4.5 (ст. 326) |
| AblationComponent | GPT-2 vs Llama | 14.7% MSE reduction: Llama-7B vs GPT-2 (12) | 4.5 (ст. 327) |
| DataPreprocessing | RevIN | Reversible instance normalization: zero mean, unit std | 3.1 (ст. 78) |
| DataPreprocessing | Channel Independence | N variables → N univariate series, processed independently | 3 (ст. 72) |
| CodeRepository | GitHub | `https://github.com/KimMeen/Time-LLM` | Abstract (ст. 27), B.1 |
| Baseline | GPT4TS (Zhou et al. 2023a) | Fine-tuned LLM for TS; главный прямой конкурент | 4.1 (ст. 121) |
| Baseline | PatchTST (Nie et al. 2023) | SOTA Transformer with patching | 4.1 (ст. 122) |
| Baseline | TimesNet (Wu et al. 2023) | Temporal 2D-variation modeling | 4.1 (ст. 122) |
| Baseline | DLinear (Zeng et al. 2023) | Simple linear model | 4.1 (ст. 122) |
| Baseline | LLMTime (Gruver et al. 2023) | Zero-shot LLM forecaster (7B comparable) | 4.4 (ст. 288) |
| Baseline | N-HiTS (Challu et al. 2023b) | Neural hierarchical interpolation; SOTA short-term | 4.2 (ст. 174) |
| Baseline | N-BEATS (Oreshkin et al. 2020) | Neural basis expansion | 4.2 Табл. 2 |
| Baseline | FEDformer, Autoformer, Informer, Reformer, ETSformer, LightTS, Stationary | Transformer-based forecasting models | 4.1 Табл. 1 |
