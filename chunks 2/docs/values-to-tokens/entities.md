# TokenCast: Извлечённые сущности (ревизия v1.2)

**article_id:** `values-to-tokens` | **Дата:** 2026-05-22
**Типы (v1.2):** А (q00), А+ (q01), В (q02), З (q03), Г (q04), Б (q05), Д (q06), Е (q07), И (q08), Ж (q09), К (q10), Л (q11)

## 3.1. Сущности Eval API

### Тип А — Способы применения LLM

| Сущность | Значение | Раздел |
|----------|----------|--------|
| **TokenCast** | LLM-driven framework: symbolic discretization → temporal tokens → unified vocabulary → cross-modality alignment → generative fine-tuning → autoregressive forecasting | Abstract, 3 |
| Time-LLM (Jin et al. 2023) | Linear adapter / soft prompt injection TS features into LLM | 1, 4.1.2 |
| GPT4TS (Zhou et al. 2023) | Fine-tune GPT-2 for TS | 4.1.2 |
| PromptCast (Xue & Salim 2023) | Soft prompts to guide frozen LLM | 2 |
| TEMPO (Cao et al. 2023) | Linear adapters to project TS into LLM semantic space | 2 |
| TaskParadigm: **Symbolic Discretization** | TS → discrete tokens via VQ codebook → unified LLM vocabulary → autoregressive generation | 3.3 |
| TaskParadigm: **Generative Fine-tuning** | Prompt-based generative fine-tuning: LLM outputs both reasoning text + future temporal tokens | 3.6 |
| TaskParadigm: **Autoregressive Alignment** | Frozen LLM + trainable shared embedding E; alignment via next-token prediction over concatenated [Z_q, Y] | 3.5 |

### Тип Г — Представление TS / Токенизация

| Сущность | Значение | Раздел |
|----------|----------|--------|
| **Decoupled Dynamic Tokenizer** | RevIN (на исторических данных) → causal encoder → VQ codebook (K embedding vectors) → nearest neighbor quantization → causal decoder → inverse RevIN | 3.3.1 |
| **VQ Codebook** | C_i = {e_{i,k}} для каждого домена i; K embedding vectors размерности d | 3.3.1 |
| **Diversity Loss** | L_diversity = (1/N)Σ 1/(d_i+ε) для предотвращения codebook collapse | 3.3.2 |
| **Unified Vocabulary** | V = V_orig ∪ {K temporal tokens} ∪ {S special tokens}; shared embedding matrix E | 3.5 |
| **Gaussian Initialization** | Новые temporal token embeddings инициализируются из N(μ, Σ) оригинальных word embeddings | 3.5 |
| **Prompt Template** | 4 компонента: domain knowledge, task instructions, statistical properties, discrete time series tokens | 3.4 |
| TS_Representation | Symbolic discretization: continuous TS → discrete codebook indices → LLM-compatible tokens | 3.3 |
| Tokenization_Method | Vector Quantization (VQ) with shared causal encoder-decoder, domain-specific codebooks, RevIN on historical only | 3.3 |

### Тип В — Датасеты

| Dataset | Domain | Horizons |
|---------|--------|----------|
| Economic | Finance | {24,36,48,60} |
| Health | Healthcare | {24,36,48,60} |
| Web | Web traffic | {24,36,48,60} |
| Stock-NY | Finance (NYSE) | {24,36,48,60} |
| Stock-NA | Finance (NASDAQ) | {24,36,48,60} |
| Nature | Environment | {24,48,96,192} |

### Тип Ж — Результаты

TokenCast — best in 5/6 datasets (MSE) и 5/6 (MAE); особенно силён на context-rich данных (Economic, Web, Stock).

## 3.2. Онтология без вопросов Eval API

| Класс | Сущность |
|-------|----------|
| BaseLLM | GPT-2 (через GPT4TS baseline), backbone LLM (не указан конкретно для TokenCast, но используется pre-trained LLM) |
| TuningStrategy | Frozen LLM + trainable embedding E (alignment), затем generative fine-tuning |
| Technique | VQ (Vector Quantization), Causal Encoder-Decoder, RevIN (historical-only), Diversity Loss, Codebook, Unified Vocabulary, Structured Prompt Template, Boundary Markers, Straight-Through Estimator |

## 3.3. Расширяемые

| Класс | Сущность |
|-------|----------|
| PromptDesign | Structured prompt template: domain knowledge + task instructions + statistics + discrete tokens |
| TrainingRegime | Cross-modality alignment (frozen LLM) → generative fine-tuning |
| AblationComponent | w/o alignment, w/o fine-tuning, w/o text context, w/o local info |
| Baseline | Time-LLM, GPT4TS, TimeDART, SimMTM, Crossformer, Autoformer, DLinear |
