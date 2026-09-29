# LLM-TS Survey: Извлечённые сущности (ревизия v1.2)

**article_id (Eval API):** `llm-ts-survey`
**Дата ревизии:** 2026-05-22
**Типы вопросов Eval API (v1.2):** А (q00), А+ (q01), В (q02), З (q03), Г (q04), Б (q05), Д (q06), Е (q07), И (q08), Л (q09)
**Основание:** переход от v1.0 (2 типа) к v1.2 (10 типов) согласно `eval-api/docs/questions.md`

---

## 3.1. Сущности, непосредственно затребованные вопросами Eval API

### Тип А — Все способы применения LLM (`llm-ts-survey_q00`)

Вопрос: *«Какие существуют способы использования LLM для прогноза временных рядов (TS), о которых идет речь в статье «Large Language Models for Time Series: A Survey»?»*

**Ключевая особенность:** обзорная статья выделяет **5 категорий** (мета-уровень таксономии), каждая из которых содержит подкатегории и представительные работы.

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| LLM_Method | **Prompting** (категория 1) | Прямой промптинг LLM числовыми рядами как текстом. Подтипы: Number-Agnostic (PromptCast) и Number-Specific (LLMTime — space-prefixed digits) | Abstract (ст. 11), 3.1 (ст. 131–167) | А |
| LLM_Method | **Time Series Quantization** (категория 2) | Дискретизация числовых рядов в дискретные токены для LLM. Подтипы: VQ-VAE (DeWave, TOTEM), K-Means (AudioLM, SpeechGPT), Text Categories (TDML) | Abstract (ст. 11), 3.2 (ст. 168–189) | А |
| LLM_Method | **Aligning** (категория 3) | Обучение отдельного TS-энкодера + выравнивание эмбеддингов TS с семантическим пространством LLM. Подтипы: Similarity Match (contrastive loss: ETP, TEST, JoLT) и LLM Backbone (GPT4TS, Time-LLM, Lag-Llama) | Abstract (ст. 11), 3.3 (ст. 191–215) | А |
| LLM_Method | **Vision as Bridge** (категория 4) | Использование визуальной модальности как моста между TS и LLM. Подходы: Paired Data (ImageBind, AnyMAL), Physics Relationships (IMUGPT), TS Plots as Images (CLIP-LSTM, Insight Miner) | Abstract (ст. 11), 3.4 (ст. 217–225) | А |
| LLM_Method | **Tool Integration** (категория 5) | LLM генерирует косвенные инструменты (код, API-вызовы) для обработки TS. Подтипы: Code (CTG++), API Call (ToolLLM), Text Domain Knowledge (SHARE, GG-LLM) | Abstract (ст. 11), 3.5 (ст. 227–235) | А |
| TaskParadigm | Number-Agnostic Prompting | TS как текст: прямое преобразование чисел в строки, промптинг «из коробки» без дообучения | 3.1 (ст. 133) | А |
| TaskParadigm | Number-Specific Tokenization | Пробельно-разделённая токенизация цифр (LLMTime), масштабирование, фиксированная точность | 3.1 (ст. 166) | А |
| TaskParadigm | VQ-VAE Quantization | Codebook-дискретизация: ближайший сосед в латентном пространстве | 3.2 (ст. 172–183) | А |
| TaskParadigm | K-Means Clustering | Индексная токенизация через центроиды кластеров | 3.2 (ст. 185) | А |
| TaskParadigm | Contrastive Alignment | Контрастное обучение: sim(g_φ(x_s), f_θ(x_t)) | 3.3 (ст. 197–211) | А |
| TaskParadigm | LLM as Frozen Backbone | Замороженная LLM как backbone после TS-энкодера (GPT4TS, Time-LLM) | 3.3 (ст. 215) | А |
| TaskParadigm | Vision-Language Bridge | VLM как посредник: TS → visual → text/LLM | 3.4 (ст. 217–225) | А |
| TaskParadigm | LLM-as-Tool-Generator | LLM → код/API → обработка TS (не прямое взаимодействие) | 3.5 (ст. 227–235) | А |

**Представительные работы (методы) для типа А:**

| LLM_Method | Представительная работа | Парадигма | Раздел |
|------------|------------------------|-----------|--------|
| PromptCast | Xue and Salim 2022 | Number-Agnostic Prompting | 3.1 |
| LLMTime | Gruver et al. 2023 | Number-Specific Tokenization | 3.1 |
| TabLLM | Hegselmann et al. 2023 | Prompting for tabular data | 3.1 |
| AuxMobLCast | Xue et al. 2022 | Prompting for POI flows | 3.1 |
| DeWave | Duan et al. 2023 | VQ-VAE Quantization | 3.2 |
| TOTEM | Talukder and Gkioxari 2023 | VQ-VAE Quantization | 3.2 |
| AudioLM | Borsos et al. 2023 | K-Means Quantization | 3.2 |
| SpeechGPT | Zhang et al. 2023a | K-Means Quantization | 3.2 |
| Chronos | Ansari et al. 2024 | Binning Quantization | 3.2 |
| TDML | Yu et al. 2023 | Text Categories | 3.2 |
| ETP | Liu et al. 2023a | Contrastive Alignment | 3.3 |
| TEST | Sun et al. 2023 | Contrastive Alignment | 3.3 |
| TENT | Zhou et al. 2023b | Contrastive Alignment | 3.3 |
| JoLT | Cai et al. 2023 | Contrastive Alignment | 3.3 |
| GPT4TS | Zhou et al. 2023a | LLM Backbone (GPT-2) | 3.3 |
| TEMPO | Cao et al. 2023 | LLM Backbone + decomposition | 3.3 |
| LLM4TS | Chang et al. 2023 | LLM Backbone + two-stage FT | 3.3 |
| UniTime | Liu et al. 2023e | LLM Backbone + domain descriptions | 3.3 |
| Time-LLM | Jin et al. 2023a | LLM Backbone (LLaMA-7B) + reprogramming | 3.3 |
| Lag-Llama | Rasul et al. 2023 | LLM Backbone (LLaMA arch.) + pretraining | 3.3 |
| ImageBind | Girdhar et al. 2023 | Vision Bridge (Paired Data) | 3.4 |
| AnyMAL | Moon et al. 2023 | Vision Bridge (Multimodal Adapter) | 3.4 |
| CLIP-LSTM | Wimmer and Rekabsaz 2023 | Vision Bridge (TS Plots) | 3.4 |
| CTG++ | Zhong et al. 2023 | Tool (Code Generation) | 3.5 |
| ToolLLM | Qin et al. 2023 | Tool (API Call) | 3.5 |
| SHARE | Zhang et al. 2023d | Tool (Domain Knowledge) | 3.5 |

### Тип В — Бенчмарки и датасеты (`llm-ts-survey_q02`)

Вопрос: *«Какие бенчмарки используются для оценки качества прогнозирования, о которых идет речь в статье «Large Language Models for Time Series: A Survey»?»*

**Специфика:** обзорная статья фокусируется на **мультимодальных** датасетах (TS + текст), поскольку именно они необходимы для применения LLM к TS. Традиционные TS-бенчмарки (ETT, Weather и т.д.) упоминаются в контексте представительных работ (GPT4TS, Time-LLM, Lag-Llama), но не каталогизируются явно.

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Benchmark | **Multimodal TS-Text Datasets** (каталог) | 10+ датасетов, организованных по доменам: IoT, Finance, Healthcare, Audio, Music, Speech | 5. Multimodal Datasets (ст. 249–355) | В |
| Dataset | Ego4D | 3670h данных, 3.85M повествований; модальности: text, IMU, video, audio, 3D; задачи: classification, forecasting | 5, Табл. 3 | В |
| Dataset | DeepSQA | 25h данных, 91K вопросов; text + IMU; classification, QA | 5, Табл. 3 | В |
| Dataset | PIXIU | 136K инструкций; text + tables; 5 NLP tasks + forecasting | 5, Табл. 3 | В |
| Dataset | MoAT | 6 датасетов, ~2K шагов; text + TS; forecasting (fuel, metal, stock, bitcoin) | 5, Табл. 3 | В |
| Dataset | Zuco 2.0 | 739 предложений; text + eye-tracking + EEG; classification, text generation | 5, Табл. 3 | В |
| Dataset | PTB-XL | 60h данных, 71 утверждение; text + ECG; classification | 5, Табл. 3 | В |
| Dataset | ECG-QA | 70 шаблонов вопросов; text + ECG; classification, QA | 5, Табл. 3 | В |
| Dataset | OpenAQA-5M | 5.6M (audio, question, answer) троек; text + audio; tagging, classification | 5, Табл. 3 | В |
| Dataset | MusicCaps | 5.5K музыкальных клипов; text + music; captioning, generation | 5, Табл. 3 | В |
| Dataset | CommonVoice | 7335 часов речи на 60 языках; text + speech; ASR, translation | 5, Табл. 3 | В |
| Dataset | Ego-Exo4D | Экспертные комментарии + narrate-and-act + atomic action descriptions | 5 (ст. 336) | В |
| Dataset | Zuco 1.0 | Eye-tracking + EEG во время чтения | 5 (ст. 339) | В |
| Dataset | AudioSet | 2M 10-секундных аудиоклипов, 527 меток | 5 (ст. 341) | В |
| Dataset | MTG-Jamendo | 55,000 аудиотреков | 5 (ст. 341) | В |
| Dataset | Libri-Light | 60,000 часов речи (English) | 5 (ст. 341) | В |
| Benchmark | **Традиционные TS-бенчмарки** (упомянуты косвенно) | ETT, Weather, Electricity, Traffic, ILI, M4 — используются в представительных работах (GPT4TS, Time-LLM, Lag-Llama), но не систематизированы в обзоре | 3.3 (ст. 215 — ссылки на работы), косвенно | В |

### Тип А+ — Наилучшие способы применения LLM (`llm-ts-survey_q01`)

Вопрос: *«Какие наилучшие способы использования LLM для прогноза TS описываются в статье...?»*

**Специфика обзора:** статья не содержит собственных ablation-экспериментов. Вместо этого раздел 4 даёт **практические руководства** по выбору категории в зависимости от ситуации (данные, модель, эффективность, оптимизация).

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| LLM_Method (best) | **Prompting** (лучший при отсутствии данных) | Zero-shot без дообучения; «when no training data is available... it is preferable to use prompting-based methods» | 4 (ст. 241) | А+ |
| LLM_Method (best) | **Aligning / Quantization** (лучший при наличии данных) | С обучением; «with adequate training data, quantization or aligning-based methods become more advantageous»; наиболее изученные категории | 4 (ст. 241) | А+ |
| LLM_Method (best) | **Vision as Bridge** (лучший при наличии визуальных представлений) | «if time series data can be interpreted or associated with visual representations, these representations can be incorporated» | 4 (ст. 241) | А+ |
| LLM_Method (best) | **Tool Integration** (для непрямого взаимодействия) | «empowers LLMs with more capabilities to manage numerical data» через код/API; off-the-shelf LLM без дообучения | 4 (ст. 247) | А+ |

### Тип З — Метрики оценки качества (`llm-ts-survey_q03`)

Вопрос: *«Какие метрики используются для оценки качества прогнозирования, о которых идет речь в статье...?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Evaluation_Metric | MSE, MAE, RMSE, MAPE | Для forecasting tasks | 5 (ст. 343) | З |
| Evaluation_Metric | Accuracy, macro-F1 | Для classification tasks | 5 (ст. 343) | З |
| Evaluation_Metric | BLEU, ROUGE, METEOR, EM | Для NLP-focused tasks (captioning, QA, translation) | 5 (ст. 343) | З |

### Тип Г — Представление временных рядов (`llm-ts-survey_q04`)

Вопрос: *«Как можно использовать LLM для представления временных рядов (TS), о которых идет речь в статье...?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| TS_Representation | Number-Agnostic Tokenization | TS как сырой текст: прямое преобразование чисел в строки, промптинг «из коробки» | 3.1 (ст. 133) | Г |
| TS_Representation | Number-Specific Tokenization | Пробельно-разделённая токенизация цифр (LLMTime), масштабирование, фиксированная точность | 3.1 (ст. 166–167) | Г |
| TS_Representation | VQ-VAE Quantization | Codebook-дискретизация: ближайший сосед в латентном пространстве (K кодовых слов) | 3.2 (ст. 172–183) | Г |
| TS_Representation | K-Means Clustering | Индексная токенизация через центроиды кластеров (AudioLM, SpeechGPT) | 3.2 (ст. 185) | Г |
| TS_Representation | Text Categories (Binning) | Категоризация числовых флуктуаций в текстовые метки (напр. «U3» = рост на 3 уровня) | 3.2 (ст. 189) | Г |
| TS_Representation | Time Series Encoder + Alignment | Обучение отдельного TS-энкодера + contrastive loss для выравнивания с текстовыми эмбеддингами | 3.3 (ст. 197–211) | Г |
| TS_Representation | Vision as Bridge | Визуальные представления TS (plots) → CLIP/LLaVA → текст/LLM | 3.4 (ст. 217–225) | Г |

### Тип Б — Способы улучшения прогноза (`llm-ts-survey_q05`)

Вопрос: *«Какие способы улучшения прогноза TS описываются в статье...?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Improvement_Method | Patching | Сегментация длинных рядов на патчи для снижения сложности и улучшения производительности | 6.3 (ст. 371) | Б |
| Improvement_Method | Down-sampling | Уменьшение частоты дискретизации перед подачей в LLM | 4 (ст. 246) | Б |
| Improvement_Method | Domain Knowledge Integration | Сезонно-трендовая декомпозиция (TEMPO), частотные признаки (FreqTST), wavelet | 6.4 (ст. 373–375) | Б |
| Improvement_Method | End-to-end Training | Совместное обучение энкодера и LLM даёт лучшие результаты, чем двухстадийное | 4 (ст. 247) | Б |
| Improvement_Method | Contrastive Alignment | Contrastive loss для точного выравнивания модальностей TS и текста | 3.3 (ст. 197–204) | Б |
| Improvement_Method | Efficient Quantization | VQ-VAE/RVQ/K-Means для компактного представления TS токенами | 3.2 (ст. 168–189) | Б |

### Тип Д — Нейросетевая архитектура / backbone (`llm-ts-survey_q06`)

Вопрос: *«Какую нейросетевую архитектуру используют для прогноза TS в качестве backbone...?»*

**Специфика обзора:** не единая архитектура, а перечень backbone-LLM, используемых в обозреваемых работах.

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| BaseLLM | GPT-2 | Backbone в GPT4TS (frozen, self-attention + positional embedding layers retained) | 3.3 (ст. 215) | Д |
| BaseLLM | LLaMA-7B | Backbone в Time-LLM (reprogramming через text prototypes) | 3.3 (ст. 215) | Д |
| BaseLLM | LLaMA (architecture) | Lag-Llama: univariate probabilistic forecaster на базе LLaMA-архитектуры | 3.3 (ст. 215) | Д |
| BaseLLM | BART | Backbone в DeWave (EEG-to-text), EEG-to-Text | 3.2 (ст. 183), 3.3 (ст. 215) | Д |
| BaseLLM | PaLM-24B | Используется для health tasks (activity recognition, stress estimate) | 3.1 (ст. 164) | Д |
| BaseLLM | GPT-4 / ChatGPT | Используется в CTG++ (code generation), SHARE, ToolLLM | 3.5 (ст. 231, 235) | Д |
| BaseLLM | CLIP | Vision-Language Model (Radford et al. 2021), используется в IMU2CLIP, CLIP-LSTM | 3.4 (ст. 221, 225) | Д |
| BaseLLM | LLaVA | Vision Language Model, используется в Insight Miner | 3.4 (ст. 225) | Д |
| BaseLLM | LLaMA-2-70B | Используется в AnyMAL с легковесным адаптером | 3.4 (ст. 221) | Д |

### Тип Е — Пайплайн и шаги прогнозирования (`llm-ts-survey_q07`)

Вопрос: *«Из каких шагов состоят пайплайны прогнозирования с использованием LLM...?»*

**Специфика обзора:** не конкретный пайплайн одного метода, а общая схема 5 стадий LLM-пайплайна, к которым привязаны категории таксономии.

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| Pipeline | LLM Pipeline (5 stages) | input text → tokenization → embedding → LLM → output | 1 (ст. 21–22) | Е |
| Workflow_Step | Input Stage → Prompting | TS как текст на входе; категория таксономии: Prompting | 1 (ст. 21–22) | Е |
| Workflow_Step | Tokenization Stage → Quantization | Дискретизация TS в специальные токены; категория: Quantization | 1 (ст. 21–22) | Е |
| Workflow_Step | Embedding Stage → Aligning | TS-энкодер + выравнивание эмбеддингов; категория: Aligning | 1 (ст. 21–22) | Е |
| Workflow_Step | LLM Stage → Vision as Bridge | Визуальные представления как мост к LLM; категория: Vision as Bridge | 1 (ст. 21–22) | Е |
| Workflow_Step | Output Stage → Tool Integration | LLM генерирует косвенные инструменты (код/API); категория: Tool | 1 (ст. 21–22) | Е |
| TaskParadigm | Формальная модель | Input: x_s ∈ R^{T×c} + x_t (text); Output: y (TS/text/numbers); Model: f_θ, g_φ, h_ψ | 2 (ст. 31–35) | Е |

### Тип И — Подготовка данных и обучение доп. параметров (`llm-ts-survey_q08`)

Вопрос: *«Какие шаги подготовки данных и обучения дополнительных параметров описываются в статье...?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| DataPreprocessing | Patching | Сегментация длинных рядов на патчи (Nie et al. 2022); улучшает производительность и снижает сложность | 6.3 (ст. 371) | И |
| DataPreprocessing | Down-sampling | Уменьшение частоты дискретизации для обработки длинных последовательностей | 4 (ст. 246) | И |
| DataPreprocessing | Scaling + Fixed Precision | Масштабирование TS для оптимизации использования токенов + фиксированная точность (напр. 2 знака) | 3.1 (ст. 166) | И |
| TrainingRegime | Zero-shot (no training) | Prompting-методы без дообучения LLM | 4 (ст. 241) | И |
| TrainingRegime | Two-stage training | VQ-VAE: сначала обучение квантователя, затем LLM; может давать субоптимальные результаты | 3.2 (ст. 172), 4 (ст. 247) | И |
| TrainingRegime | End-to-end training | Aligning-методы: совместное обучение TS-энкодера и LLM | 3.3, 4 (ст. 247) | И |
| TrainingRegime | Off-the-shelf LLM (no fine-tuning) | Tool Integration: применение готовых LLM (GPT-4) без дообучения | 4 (ст. 247) | И |

### Тип Л — Нормировка / преобразование TS (`llm-ts-survey_q09`)

Вопрос: *«Какие виды преобразования временных рядов для нормировки описываются в статье...?»*

| Класс | Сущность | Значение | Раздел-источник | Тип вопроса |
|-------|----------|----------|-----------------|-------------|
| DataPreprocessing (normalization) | Scaling (LLMTime) | Масштабирование TS для оптимизации использования токенов | 3.1 (ст. 166) | Л |
| DataPreprocessing (normalization) | Fixed Precision | Фиксированная точность (напр. 2 знака после запятой) для эффективного управления длиной контекста | 3.1 (ст. 166) | Л |
| DataPreprocessing (normalization) | Space-Prefixed Digit Tokenization | Разделение цифр пробелами для корректной BPE-токенизации чисел | 3.1 (ст. 166) | Л |
| DataPreprocessing (normalization) | VQ-VAE Quantization | Дискретизация непрерывных TS в индексы кодовой книги | 3.2 (ст. 172–183) | Л |
| DataPreprocessing (normalization) | Binning (Chronos) | Квантование вещественных TS в дискретные бины | 3.2 (ст. 187) | Л |
| DataPreprocessing (normalization) | Text Categories (TDML) | Преобразование ценовых флуктуаций в текстовые категории («D1»–«U5+») | 3.2 (ст. 189) | Л |
| DataPreprocessing (normalization) | Patching | Сегментация TS на патчи перед LLM (снижает размерность) | 4 (ст. 246), 6.3 (ст. 371) | Л |
| DataPreprocessing (normalization) | Seasonal-Trend Decomposition | Декомпозиция на сезонную, трендовую и остаточную компоненты (TEMPO) | 6.4 (ст. 375) | Л |

---

## 3.2. Сущности из онтологии без прямых вопросов Eval API

| Класс | Сущность | Значение | Раздел-источник |
|-------|----------|----------|-----------------|
| BaseLLM | GPT-2 | Backbone в GPT4TS | 3.3 (ст. 215) |
| BaseLLM | LLaMA-7B | Backbone в Time-LLM | 3.3 (ст. 215) |
| BaseLLM | LLaMA-2-70B | Используется в AnyMAL | 3.4 (ст. 221) |
| BaseLLM | PaLM-24B | Используется для health tasks у Liu et al. | 3.1 (ст. 164) |
| BaseLLM | BART | Backbone в DeWave, EEG-to-Text | 3.2 (ст. 183), 3.3 (ст. 215) |
| BaseLLM | GPT-4 / ChatGPT | Используется в CTG++, SHARE, ToolLLM | 3.5 (ст. 231, 235) |
| BaseLLM | LLaVA | Vision Language Model в Insight Miner | 3.4 (ст. 225) |
| BaseLLM | CLIP | Vision-Language Model (Radford et al. 2021) | 3.4 (ст. 221, 225) |
| Technique | BPE Tokenization | Штатная токенизация LLM (проблематична для чисел) | 3.1 (ст. 166) |
| Technique | Space-Prefixed Digit Tokenization | Разделение цифр пробелами для корректной токенизации чисел | 3.1 (ст. 166) |
| Technique | VQ-VAE (Vector Quantized VAE) | Codebook-дискретизация: K кодовых слов размерности D | 3.2 (ст. 172) |
| Technique | RVQ (Residual Vector Quantization) | Иерархия множественных векторных квантователей | 3.2 (ст. 183) |
| Technique | Patching (Nie et al. 2022) | Сегментация TS на патчи перед LLM | 3.3 (ст. 215), 6.3 (ст. 371) |
| Technique | Seasonal-Trend Decomposition | Декомпозиция TS (TEMPO) | 3.3 (ст. 215), 6.4 (ст. 375) |
| Technique | Contrastive Learning | Contrastive loss для выравнивания модальностей | 3.3 (ст. 197–204) |
| Technique | Q-Former (Querying Transformer) | Используется в JoLT для alignment | 3.3 (ст. 211) |
| Technique | Canonical Correlation Analysis | Метод alignment (MATM) | 3.3 (ст. 213) |
| Technique | Optimal Transport | Функция потерь для alignment (ECG-LLM) | 3.3 (ст. 213) |
| Domain | IoT | Ego4D, DeepSQA | 5, Табл. 3 |
| Domain | Finance | PIXIU, MoAT | 5, Табл. 3 |
| Domain | Healthcare | Zuco, PTB-XL, ECG-QA | 5, Табл. 3 |
| Domain | Audio | OpenAQA-5M, AudioSet | 5, Табл. 3 |
| Domain | Music | MusicCaps, MTG-Jamendo | 5, Табл. 3 |
| Domain | Speech | CommonVoice, Libri-Light | 5, Табл. 3 |

---

## 3.3. Потенциально-расширяемые классы

| Класс | Сущность | Значение | Раздел-источник |
|-------|----------|----------|-----------------|
| PromptDesign | PromptCast template | «From {t1} to {t_obs}, the average temperature of region {U_m} was {x_t^m} degree...» | 3.1, Табл. 1 |
| PromptDesign | LLMTime tokenization | «0.123, 1.23, 12.3, 123.0 → 1 2 , 1 2 3 , 1 2 3 0 , 1 2 3 0 0» | 3.1, Табл. 1 |
| TrainingRegime | Zero-shot (no training data) | Prompting-методы: применение LLM без дообучения | 4 (ст. 241) |
| TrainingRegime | Two-stage training | VQ-VAE: сначала обучение квантователя, затем LLM | 3.2 (ст. 172), 4 (ст. 247) |
| TrainingRegime | End-to-end training | Aligning-методы: совместное обучение энкодера и LLM | 3.3, 4 (ст. 247) |
| ComputationalCost | Billion-parameter vs million-parameter | Prompting/Tool → billion-param; Aligning/Quantization → million to billion | 4 (ст. 243) |
| DataPreprocessing | Patching | Сегментация длинных рядов на патчи для эффективности | 6.3 (ст. 371) |
| DataPreprocessing | Down-sampling | Уменьшение частоты перед подачей в LLM | 4 (ст. 246) |
| CodeRepository | GitHub (awesome-llm-time-series) | `https://github.com/xiyuanzh/awesome-llm-time-series` | Abstract (ст. 12), сноска 1 |
| Baseline | GPT4TS | Один из наиболее цитируемых LLM-for-TS методов в обзоре | 3.3 |
| Baseline | Time-LLM | Reprogramming LLaMA-7B | 3.3 |
| Baseline | Lag-Llama | Foundation model (LLaMA architecture from scratch) | 3.3 |
| Baseline | LLMTime | Zero-shot forecasting через number-specific tokenization | 3.1 |
