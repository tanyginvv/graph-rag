# LLM-Agents (DCATS): Структура + Сущности

**article_id (Eval API):** `llm-agents` | **Типы:** А (`llm-agents_q01`), Ж (`llm-agents_q02`)

## Структура

| Раздел | Строки |
|--------|--------|
| Abstract | 12 |
| 1. Introduction | 23–39 |
| 2. Related Work | 41–44 |
| 3. Methodology (4 компонента: dataset, metadata, LLM-agent, forecasting module) | 45–52+ |
| 3.1 Initial Proposal (5-section prompt) | 54–70 |
| Experiments (traffic volume, 6% error reduction) | далее |

## Тип А — Способы применения LLM

| Сущность | Значение |
|----------|----------|
| **DCATS** (Data-Centric Agent for TS) | LLM-agent для AutoML на TS: reasoning over metadata → proposal generation → iterative refinement → data selection → forecasting |
| TaskParadigm: **LLM-as-AutoML-Agent** | LLM не обрабатывает TS напрямую, а планирует data-centric pipeline (Tool Integration из taxonomy llm-ts-survey) |
| AIDE (general AutoML LLM-agent) | Baseline/аналог из литературы |

## Тип Ж — Задача, горизонт, улучшение

| Сущность | Значение |
|----------|----------|
| Task_Type | Traffic volume forecasting (spatial-temporal) |
| Improvement_Percent | **6% error reduction** across all tested models and time horizons |
| Forecast_Horizon | Multiple horizons tested (не специфицированы в Abstract) |
| Evaluation_Metric | Forecasting error (MAE/MSE — implied) |

## 3.2

- **BaseLLM**: LLM-agent (используется для reasoning, не как backbone forecaster)
- **Technique**: Prompt Engineering (5-section: background, task, guidelines, neighbors, output), Iterative Proposal Refinement, Metadata-driven Data Selection

## Ontology Gap

- **LLM-as-Agent** paradigm: LLM не предсказывает TS, а планирует data pipeline. Соответствует "Tool Integration" категории из llm-ts-survey
- **Data-Centric AutoML**: специфический подтип Tool Integration

## Chunking Plan

| ID | Раздел | Типы | Приоритет |
|----|--------|------|-----------|
| C01 | Abstract | А, Ж | Высокий |
| C02 | 1. Introduction (DCATS data-centric approach) | А | Высокий |
| C03 | 3. Methodology (4 components + iterative process) | А | Высокий |
| C04 | Experiments (6% improvement, traffic forecasting) | Ж | Высокий |
