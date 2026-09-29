# Values-to-Tokens (TokenCast): Структурная разметка

**article_id (Eval API):** `values-to-tokens` | **Файл:** `FROM VALUES TO TOKENS...md` | **Строк:** 607

## Иерархия разделов

| Уровень | Номер | Название | Строки |
|---------|-------|----------|--------|
| 0 | — | Abstract | 9 |
| 1 | 1 | Introduction | 11–26 |
| 1 | 2 | Related Work | 27–32 |
| 1 | 3 | The Proposed TokenCast | 33–106 |
| 2 | 3.1 | Problem Formulation | 37–39 |
| 2 | 3.2 | Framework Overview (3 стадии) | 41–45 |
| 2 | 3.3 | Time Series Discretization (Tokenizer + Codebook) | 47–74 |
| 2 | 3.4 | Pre-trained LLM Backbone Formulation | 76–78 |
| 2 | 3.5 | Cross-Modality Alignment | 80–99 |
| 2 | 3.6 | Generative Fine-tuning & Forecasting | 101–106 |
| 1 | 4 | Experiments | 107–180+ |
| 2 | 4.1 | Experimental Setup (Datasets, Baselines, Implementation) | 111–162 |
| 2 | 4.2 | Forecasting Performance Analysis | 164–167 |
| 2 | 4.3 | Ablation Studies | 168–180+ |
| 1 | — | References + Appendix | конец |

## Ключевые блоки

| Блок | Разделы | Типы вопросов |
|------|---------|---------------|
| Abstract | Abstract | А, Г |
| Introduction (symbolic discretization идея) | 1 | А |
| Time Series Tokenizer (VQ + codebook + RevIN) | 3.3 | Г |
| LLM Backbone + Prompt Template | 3.4 | А |
| Cross-Modality Alignment (unified vocabulary) | 3.5 | А, Г |
| Generative Fine-tuning | 3.6 | А |
| Datasets (Табл. 1) | 4.1.1, Табл. 1 | В |
