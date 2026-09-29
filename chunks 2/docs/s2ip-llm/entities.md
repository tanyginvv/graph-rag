# S2IP-LLM: Извлечённые сущности (ревизия v1.2)

**article_id:** `s2ip-llm` | **Дата:** 2026-05-22
**Типы (v1.2):** А (q00), А+ (q01), В (q02), З (q03), Г (q04), Б (q05), Д (q06), Е (q07), И (q08), Ж (q09), К (q10), Л (q11)

## 3.1. Сущности Eval API

### Тип А — Способы применения LLM (`s2ip-llm_q00`)

| Сущность | Значение | Раздел |
|----------|----------|--------|
| **S²IP-LLM** | Semantic Space Informed Prompt learning: decomposition (trend+seasonal+residual) → patching → concatenation → alignment with semantic anchors → prefix-prompts → GPT-2 | Abstract, 3 |
| Time-LLM (Jin et al. 2024) | Reprogramming TS via text prototypes; frozen LLM; baseline | 1, 2.2, 4 |
| OFA/OneFitsAll (Zhou et al. 2023) | GPT-2 fine-tuned for TS (partial tuning); baseline | 1, 2.2, 4 |
| TEST (Sun et al. 2023) | Text prototype aligned embedding → activate LLM | 2.2 |
| TaskParadigm: **Semantic Space Prompting** | Pre-trained word token embeddings → semantic anchors E' → top-K alignment via cosine similarity → prefix-prompts | 3.3 |
| TaskParadigm: **Decomposition-based Tokenization** | Additive trend/seasonal/residual decomposition → individual patching → concatenation into meta-token | 3.2 |

### Тип Д — Нейросетевая архитектура / Backbone

| Сущность | Значение | Раздел |
|----------|----------|--------|
| **Backbone: GPT-2** | GPT-2 (Radford et al. 2019) как основной LLM. Параметры multi-head attention и FFN заморожены; fine-tune только positional embedding layer и layer normalization layers | 3.5 |
| Time-LLM(L) | LLaMA backbone вариант Time-LLM (baseline) | 4.1, Табл. 1 |
| Time-LLM(G) | GPT-2 backbone вариант Time-LLM (baseline) | 4.1, Табл. 1 |
| Tokenization Module | RevIN → additive seasonal-trend decomposition → overlapping patching (N_P patches of L_P) → concatenation [tre, sea, res] → projection layer g(·) → D-dimensional TS embedding | 3.2 |
| Semantic Anchors | E' = f(E) ∈ R^{V'×D}: reduced set из V=50257 токенов GPT-2 через generic mapping function f(·) | 3.3 |
| Cosine Similarity Alignment | γ(P, e'_m) = (P·e'_m)/(‖P‖‖e'_m‖); top-K selection | 3.3 |
| Output: Additive Reconstruction | Y_out = [Y_tre, Y_sea, Y_res] → Ý = Y_tre+Y_sea+Y_res (обратная декомпозиции) | 3.4 |

### Тип В — Датасеты

| Dataset | Особенности |
|---------|------------|
| Weather, Electricity, Traffic, ETTh1/ETTh2/ETTm1/ETTm2 | Long-term forecasting; input=512, horizons={96,192,336,720} |
| M4 (Yearly/Quarterly/Monthly/Weekly/Daily/Hourly) | Short-term; horizons [6,48] |

### Тип Ж — Результаты

| Метрика | S²IP-LLM превосходит baselines на большинстве датасетов; особенно в few-shot (10%/5% данных) |
|---------|-----------|
| vs Time-LLM(L) | Превосходит на большинстве датасетов (MSE/MAE) |
| vs OFA | Значительно лучше на short-term M4 |

## 3.2. Онтология без вопросов Eval API

| Класс | Сущность |
|-------|----------|
| BaseLLM | GPT-2 (Radford et al. 2019), LLaMA (Touvron et al. 2023 — через Time-LLM baseline) |
| TuningStrategy | Partial Fine-tuning: frozen attention+FFN, trainable: positional embedding + layer norm |
| Technique | RevIN, Additive Seasonal-Trend Decomposition, Overlapping Patching, Meta-token Concatenation, Semantic Anchor Mapping (f(·)), Cosine Similarity Alignment, Top-K Prompt Retrieval, Additive Reconstruction |
| Domain | Energy (ETT, Electricity), Weather, Transport (Traffic), Mixed (M4) |

## 3.3. Расширяемые

| Класс | Сущность |
|-------|----------|
| PromptDesign | Semantic anchors как soft prompts (не текстовые шаблоны) |
| TrainingRegime | Few-shot (10%, 5%), Long-term, Short-term |
| AblationComponent | Prompt length K∈{2,4,8,16,32}; λ alignment strength; V' semantic anchors count; decomposition on/off |
| Baseline | Time-LLM(L), Time-LLM(G), OFA, iTransformer, PatchTST, DLinear, FEDformer, Autoformer, TimesNet, ETSformer, Informer, LightTS, N-HiTS, N-BEATS |
