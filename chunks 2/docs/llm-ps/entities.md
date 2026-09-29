# LLM-PS: Извлечённые сущности (ревизия v1.2)

**article_id (Eval API):** `llm-ps` | **Дата ревизии:** 2026-05-22
**Типы вопросов (v1.2):** А (q00), А+ (q01), В (q02), З (q03), Г (q04), Б (q05), Д (q06), Е (q07), И (q08), Ж (q09), К (q10), Л (q11)

## 3.1. Сущности Eval API

### Тип А — Способы применения LLM (`llm-ps_q00`)

| Сущность | Значение | Раздел |
|----------|----------|--------|
| **LLM-PS** (основной метод) | LLM fine-tuning framework: MSCNN (multi-scale patterns) + T2T (semantics extraction) → GPT-2 backbone; LoRA parameter-efficient | Abstract, 3 |
| TimeLLM (Jin et al. 2024) | Reprogramming TS + text prototypes + PaP → frozen LLM | 1, 2.1 |
| CALF (Liu et al. 2024a) | Cross-modal fine-tuning: separate TS/text branches + alignment | 1, 2.1 |
| GPT4TS / OFA (Zhou et al. 2023) | GPT-2 fine-tuned for TS: patching + frozen backbone with tuned heads | 1, 2.1 |
| PromptCast (Xue & Salim 2023) | TS + text → prompts → LLM prediction | 2.1 |
| LLM-TS (Chen et al. 2024) | CNN branch + LLM guidance via mutual information minimization | 2.1 |
| LLMMixer (Kowsher et al. 2024) | Multi-scale mixing in LLMs for TS | 4.4 |
| TaskParadigm: **LoRA Fine-tuning** | Parameter-efficient LLM adaptation (rank=8, scale=32, dropout=0.1) | 3.4 |
| TaskParadigm: **Patterns + Semantics** | В отличие от alignment-only методов, LLM-PS извлекает temporal patterns и semantics из TS перед LLM | 1, 3 |

### Тип Б — Способы улучшения (`llm-ps_q05`)

| Сущность | Значение | Раздел |
|----------|----------|--------|
| **MSCNN** (Multi-Scale CNN) | Иерархические bottleneck блоки с параллельными ветвями; захват short-term (периодические флуктуации) и long-term (тренды) паттернов | 3.1 |
| **Wavelet Transform Decoupling** | DB4 wavelet: low-frequency → long-term patterns; high-frequency → short-term patterns; global-to-local + local-to-global assembling | 3.2 |
| **T2T (Time-to-Text)** | Masked autoencoding (75% masking): реконструкция патчей + предсказание семантических меток через similarity с LLM embeddings | 3.3 |
| **LoRA** (Hu et al. 2021) | Low-Rank Adaptation: rank=8, scale=32, dropout=0.1; обучаются только low-rank матрицы | 3.4 |
| **Feature Transfer (L_FEAT)** | Выравнивание MSCNN признаков F_MS и T2T признаков F_T2T через MSE | 3.4 |
| **Semantic Filtering** | Фильтрация LLM word embeddings: top-100 релевантных TSF слов через cosine similarity | B.4 |

### Тип В — Датасеты

| Dataset | Особенности | Раздел |
|---------|------------|--------|
| ETTh1, ETTh2, ETTm1, ETTm2 | Electricity Transformer Temperature; 7 переменных; hourly/15-min | 4.1, A.2 |
| Weather | 21 метеоиндикаторов, 10-min, 2020 | 4.1, A.2 |
| Electricity | 321 переменная, энергопотребление, hourly, 2017 | 4.1, A.2 |
| Traffic | 862 сенсора, 15-min, 2016–2018 | 4.1, A.2 |
| ILI | Influenza-like illness, weekly, 7 переменных, 2002–2021 | 4.1, A.2 |
| ECG | 48 записей, 2-channel, 360 Hz, biomedical | 4.1, A.2 |
| M4 | 100K рядов: yearly, quarterly, monthly, others | 4.2, A.2 |

### Тип Ж — Результаты

| Улучшение | Значение |
|-----------|----------|
| vs CALF (long-term) | 6% MSE, 3% MAE reduction |
| vs TimeLLM (long-term) | 11% MSE, 9% MAE |
| vs GPT4TS (long-term) | 9% MSE, 5% MAE |
| vs CALF (few-shot) | 3% improvement |
| vs TimeLLM (few-shot) | 17% improvement |
| vs GPT4TS (few-shot) | 13% improvement |
| Short-term SOTA | SMAPE 11.721, MASE 1.561, OWA 0.840 — best overall |
| Speed vs LLMMixer | 17% better MSE, 9% of training time |

### Тип З — Метрики

MSE, MAE, MSAE (Mean Absolute Scaled Error), SMAPE, OWA.

## 3.2. Онтология без вопросов Eval API

| Класс | Сущность | Раздел |
|-------|----------|--------|
| BaseLLM | GPT-2 (first 6 layers) | 4 (Implementation) |
| TuningStrategy | LoRA (rank=8, scale=32, dropout=0.1); backbone partially frozen | 3.4 |
| Technique | MSCNN (bottleneck blocks, multi-branch), Wavelet Transform (DB4), T2T (masked autoencoder), Semantic Filtering, Feature Transfer | 3.1–3.4 |
| Domain | Energy (ETT, Electricity), Weather, Transport (Traffic), Health (ILI, ECG), Finance (M4 subsets) | A.2 |

## 3.3. Расширяемые

| Класс | Сущность |
|-------|----------|
| TrainingRegime | Full fine-tuning (LoRA), Few-shot (10% data), Zero-shot (cross-dataset transfer) |
| ComputationalCost | 192s (ETTh1) — 1092s (Traffic) на RTX 4090; LoRA rank=8 |
| AblationComponent | w/o T2T: 0.426 vs 0.418 MSE на ETTh1; wavelet vs Fourier vs pooling |
| DataPreprocessing | Patching (L=24), RevIN normalization |
| CodeRepository | — (не указан) |
| Baseline | CALF, TimeLLM, GPT4TS, PatchTST, iTransformer, Crossformer, FEDformer, Autoformer, TimesNet, DLinear, TiDE, N-HiTS, N-BEATS и др. |
